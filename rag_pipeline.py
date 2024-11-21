from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore, Node, NodeRelationship
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.core import QueryBundle, PromptTemplate, get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer, SimilarityPostprocessor
from prompts import LLM_CONTEXT_PROMPT
import asyncio

# Constants
RETRIEVER_SIMILARITY_TOP_K = 10

def log_step(step: str, message: str, nodes: list[NodeWithScore] = None):
    """Helper function for consistent logging"""
    print(f"[{step}] {message}")
    if nodes:
        print(f"[{step}] {'Score':^8} | Content")
        print(f"[{step}] {'-'*8}-+-{'-'*50}")
        for node in nodes:
            content = node.node.get_content()[:50].replace('\n', ' ')
            print(f"[{step}] {float(node.score):^8.2f} | {content}")

class RetrieverEvent(Event):
    """Event for the retriever step"""
    nodes: list[NodeWithScore]

class PostRetrievalTransform(Event):
    """Event for the post-retrieval transform step"""
    nodes_transformed: list[NodeWithScore]

class RAGWorkflow(Workflow):
    """Workflow for the RAG pipeline"""
    def __init__(self, llm, embed_model):
        super().__init__()
        self.llm = llm
        self.embed_model = embed_model
    
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG, triggered by a StartEvent with `query`."""
        
        query = ev.get("query")
        index_text = ev.get("index_text")
        index_keywords = ev.get("index_keywords")
        
        log_step("query", f"Querying the database with: {query}...")

        if not query:
            return None

        # store the query and index in the global context
        await ctx.set("query", query)
        await ctx.set("index_text", index_text)
        await ctx.set("index_keywords", index_keywords)

        # get the index from the global context
        if index_text is None:
            print("[errors] Index is empty, load some documents before querying!")
            return None
    
        # Combine retriever initialization and execution
        retrievers = {
            'text': index_text.as_retriever(
                similarity_top_k=RETRIEVER_SIMILARITY_TOP_K
            ),
            'keywords': index_keywords.as_retriever(
                similarity_top_k=RETRIEVER_SIMILARITY_TOP_K
            )
        }
        
        # Run retrievers concurrently
        nodes = await asyncio.gather(
            retrievers['text'].aretrieve(query),
            retrievers['keywords'].aretrieve(query)
        )
        text_nodes, keyword_nodes = nodes
        
        log_step("retrieval", "Retrieved nodes:", text_nodes)
        log_step("retrieval", "Retrieved keyword nodes:", keyword_nodes)
        
        # Create score lookup for faster matching
        text_node_map = {node.node.node_id: node for node in text_nodes}
        parent_nodes = []
        
        # Process keyword nodes more efficiently
        for kw_node in keyword_nodes:
            parent_id = kw_node.node.relationships[NodeRelationship.PARENT].node_id
            
            # Update existing text node score if found
            if parent_id in text_node_map:
                text_node_map[parent_id].score = kw_node.score
                continue
                
            # Add new parent node if not found
            try:
                parent_node = index_text.vector_store.get_nodes([parent_id])[0]
                parent_nodes.append(NodeWithScore(
                    node=parent_node,
                    score=kw_node.score
                ))
            except ValueError as e:
                print(f"[errors] Node not found: {e}")
        
        # Combine unique nodes efficiently
        all_nodes = list(text_node_map.values()) + parent_nodes
        
        log_step("retrieval", "Final combined nodes:", all_nodes)
        
        return RetrieverEvent(nodes=all_nodes)
    
    @step
    async def post_retrieval_transform(self, ctx: Context, ev: RetrieverEvent) -> PostRetrievalTransform:
        """Transform the retrieved nodes using sentence embedding optimization."""
        log_step("processing", "Starting post retrieval transform")
        
        # Create query bundle once
        query_bundle = QueryBundle(query_str=await ctx.get("query", default=None))
        
        # Apply the similarity postprocessor to remove low-scoring nodes
        similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.15
        )
        nodes_transformed = similarity_postprocessor.postprocess_nodes(nodes=ev.nodes, query_bundle=query_bundle)
        
        log_step("processing", "Similarity postprocessor done")
        
        # Initialize reranker with optimal settings
        sentence_embedding_optimizer = SentenceEmbeddingOptimizer(
            embed_model=Settings.embed_model,
            percentile_cutoff=0.7,  # Keep top 70% of sentences
            context_before=2,
            context_after=2,
            threshold_cutoff=0.15
        )
      
        # Running this takes the longest time in the pipeline
        nodes_transformed = sentence_embedding_optimizer.postprocess_nodes(nodes=nodes_transformed, query_bundle=query_bundle)
        
        log_step("processing", "Filtered and transformed nodes:", nodes_transformed)
        return PostRetrievalTransform(nodes_transformed=nodes_transformed)
        
    @step
    async def synthesize(self, ctx: Context, ev: PostRetrievalTransform) -> StopEvent:
        """Generate a streaming response from transformed nodes."""
        log_step("response", "Starting response synthesis")
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=Settings.llm,
            streaming=True,
            verbose=True,
            summary_template=PromptTemplate(LLM_CONTEXT_PROMPT),
        )
        
        response = await response_synthesizer.asynthesize(
            await ctx.get("query", default=None), 
            nodes=ev.nodes_transformed
        )
        
        log_step("response", "Response synthesis complete")
        
        return StopEvent(result=response)