from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

from llama_index.core import PromptTemplate
from prompts import QUERY_GEN_PROMPT

'''
- LLM and embeddings are set in the Settings so I shouldn't need to set them here

'''

QUERY_GEN_PROMPT_TEMPLATE = PromptTemplate(QUERY_GEN_PROMPT)

RETRIEVER_SIMILARITY_TOP_K = 10
FUSION_SIMILARITY_TOP_K = 5

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]
    
class PostRetrievalTransform(Event):
    """Result of running post retrieval transform"""

    nodes_transformed: list[NodeWithScore]
    
class ProgressEvent(Event):
    msg: str

class RAGWorkflow(Workflow):
    
    def __init__(self, llm, embed_model):
        super().__init__()
        
        # initialize the LLM and embeddings
        self.llm = llm
        self.embed_model = embed_model
    
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        
        query = ev.get("query")
        index = ev.get("index")
        
        ctx.write_event_to_stream(ProgressEvent(msg=f"Query the database with: {query}"))

        if not query:
            return None

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None
        
        # define the retrievers
        # We want to retrieve more nodes then less so set the k value high. We can filter out
        # nodes in post-processing
        vector_retriever = index.as_retriever(similarity_top_k=RETRIEVER_SIMILARITY_TOP_K)
        bm25_retriever = BM25Retriever.from_defaults(nodes=index.vector_store.get_nodes(), similarity_top_k=RETRIEVER_SIMILARITY_TOP_K)
        
        # have the retriever output progress events to the stream
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=FUSION_SIMILARITY_TOP_K,
            num_queries=3,  # Set to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            llm=self.llm,
        #    query_gen_prompt=QUERY_GEN_PROMPT_TEMPLATE
        )

        nodes = await retriever.aretrieve(query)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Retrieved {len(nodes)} nodes."))
        return RetrieverEvent(nodes=nodes)
    
    @step
    async def post_retrieval_transform(self, ctx: Context, ev: RetrieverEvent) -> PostRetrievalTransform:
        """Transform the retrieved node before synthesizing"""

        ctx.write_event_to_stream(ProgressEvent(msg=f"Transforming {len(ev.nodes)} nodes."))

        #similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        
        sentence_embedding_postprocessor = SentenceEmbeddingOptimizer(
            embed_model=self.embed_model,
            #percentile_cutoff=0.5,
            context_before=2,
            context_after=2,
            threshold_cutoff=0.7
        )
        nodes = ev.nodes
        #nodes_transformed = similarity_postprocessor.postprocess_nodes(nodes=nodes)
        #print(f"First Transform: {len(nodes_transformed)} nodes.")

        nodes_transformed = sentence_embedding_postprocessor.postprocess_nodes(nodes=nodes)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Transformed {len(nodes_transformed)} nodes."))
        
        return PostRetrievalTransform(nodes_transformed=nodes_transformed)
        

    @step
    async def synthesize(self, ctx: Context, ev: PostRetrievalTransform) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        
        ctx.write_event_to_stream(ProgressEvent(msg="Synthesizing the response"))
        
        prompt = """When possible, group output based on common fields such as Employer or Project Name and make sure to associate the response with the projects they were extracted from"""
        
        qa_prompt_tmpl = ('''
            Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information and not prior knowledge, 
            answer the query.\n
            Please synthesize the information by grouping details based on common fields, 
            such as 'Employer' and 'Project Name.' For each section, indicate the specific 
            project the information originated from. Aim to present a structured summary, 
            with grouped information under each relevant field.\n
            Query: {query_str}\n
            Answer:'''
        )
        
        qa_prompt_tmpl_2 = ('''
            Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information and not prior knowledge, 
            answer the query.\n
            Please synthesize the information first answering the query, then provide a summary of the information grouped by common fields.\n
            Query: {query_str}\n
            Answer:'''
        )
        
        qa_prompt = PromptTemplate(qa_prompt_tmpl)
        qa_prompt_tmpl_2 = PromptTemplate(qa_prompt_tmpl_2)

        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            llm=self.llm,
            streaming=True,
            verbose=True,
            summary_template=qa_prompt_tmpl_2,
            )
        
        query = await ctx.get("query", default=None)
        nodes = ev.nodes_transformed
        response = await response_synthesizer.asynthesize(query, nodes=nodes)
        ctx.write_event_to_stream(ProgressEvent(msg="Synthesized the response"))
        return StopEvent(result=response)