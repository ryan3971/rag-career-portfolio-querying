from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core import QueryBundle

from llama_index.core import PromptTemplate
from prompts import QUERY_GEN_PROMPT, LLM_CONTEXT_PROMPT, Q_A_PROMPT

'''
- LLM and embeddings are set in the Settings so I shouldn't need to set them here

'''

QUERY_GEN_PROMPT_TEMPLATE = PromptTemplate(QUERY_GEN_PROMPT)

RETRIEVER_SIMILARITY_TOP_K = 10
FUSION_SIMILARITY_TOP_K = 5

# define a query type enum

from enum import Enum
from pydantic import BaseModel


from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage

from rag_tools import get_retriever, get_post_retrieval_transform, get_synthesizer
from IPython.display import display, HTML
import pandas as pd
    
from llama_index.core.schema import Node, NodeRelationship, RelatedNodeInfo

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]
    
class PostRetrievalTransform(Event):
    """Result of running post retrieval transform"""

    nodes_transformed: list[NodeWithScore]

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
        index_text = ev.get("index_text")
        index_keywords = ev.get("index_keywords")
        config = ev.get("config")
        
        print(f"Querying the database with: {query}...")

        if not query:
            return None

        # store the query and index in the global context
        await ctx.set("query", query)
        await ctx.set("index_text", index_text)
        await ctx.set("index_keywords", index_keywords)
        await ctx.set("config", config)

        # get the index from the global context
        if index_text is None:
            print("Index is empty, load some documents before querying!")
            return None
        
        # have the retriever output progress events to the stream
        text_retriever = index_text.as_retriever(
            similarity_threshold=0.6,
            similarity_top_k=10
            )
        
        text_nodes = await text_retriever.aretrieve(query)
        
        # print the first 100 characters of the retrieved text nodes, along with their score
        # for node in text_nodes:
        #     print(f"\n{'='*80}")
        #     print(f"Text Node Content:\n{node.text[:100]}")
        #     print(f"Text Node Score: {node.score if hasattr(node, 'score') else 'N/A'}")
        
        # run the retriever on the keyword index
        retriever_keywords = index_keywords.as_retriever(
            similarity_threshold=0.6,
            similarity_top_k=10
            )
        
        keyword_nodes = await retriever_keywords.aretrieve(query)
        
        # print the first 100 characters of the retrieved keyword nodes, along with their score
        # for node in keyword_nodes:
        #     print(f"\n{'='*80}")
        #     print(f"Keyword Node Content:\n{node.text[:100]}")
        #     print(f"Keyword Node Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                
        parent_nodes = []
        for kw_node in keyword_nodes:
            parent_id = kw_node.node.relationships[NodeRelationship.PARENT].node_id
            print(f"Parent ID: {parent_id}")
            
            # get the parent node from the text index
            try:
                parent_node = index_text.vector_store.get_nodes([parent_id])[0]
                print(f"Found node: {parent_node}")
            except ValueError as e:
                print(f"Node not found: {e}")
            
            if parent_node:
                parent_nodes.append(NodeWithScore(
                    node=parent_node,
                    score=kw_node.score
                ))
                print(f"Parent node found: {parent_node.text[:100]}")
        
        all_nodes = []
        seen_ids = set()
        
        for node in text_nodes + parent_nodes:
            if node.node_id not in seen_ids:
                all_nodes.append(node)
                seen_ids.add(node.node_id)
        
        # print the first 100 characters of the first 10 nodes
        # for node in all_nodes[:10]:
        #     print(f"\n{'='*80}")
        #     print(f"Node Content:\n{node.text[:100]}")
        #     print(f"{'='*80}")
        
        visualize_retrieved_nodes(all_nodes)
                
        return RetrieverEvent(nodes=all_nodes)
    
    @step
    async def post_retrieval_transform(self, ctx: Context, ev: RetrieverEvent) -> PostRetrievalTransform:
        """Transform the retrieved node before synthesizing"""
        
        query = await ctx.get("query", default=None)
        query_bundle = QueryBundle(query_str=query)
        
        config = await ctx.get("config", default=None)
        
        postprocessor = get_post_retrieval_transform(
            post_retrieval_transform_type=config["postprocessor"]
        )
      
        nodes = ev.nodes
        nodes_transformed = postprocessor.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)

        return PostRetrievalTransform(nodes_transformed=nodes_transformed)
        
    @step
    async def synthesize(self, ctx: Context, ev: PostRetrievalTransform) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        
        response_mode_options = {"tree_summarize": ResponseMode.TREE_SUMMARIZE,
                                 "compact": ResponseMode.COMPACT,
                                 "no_text": ResponseMode.NO_TEXT}
        
        config = await ctx.get("config", default=None)
        
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
        
        q_a_prompt = PromptTemplate(Q_A_PROMPT)
        
        qa_prompt = PromptTemplate(qa_prompt_tmpl)
        qa_prompt_tmpl_2 = PromptTemplate(qa_prompt_tmpl_2)
        
        context_prompt = PromptTemplate(LLM_CONTEXT_PROMPT)

        response_synthesizer = get_synthesizer(
            response_mode_options[config["response_mode"]],
            context_prompt
            )
        
        query = await ctx.get("query", default=None)
        nodes = ev.nodes_transformed
        response = await response_synthesizer.asynthesize(query, nodes=nodes)
        print("Response generated.")
        return StopEvent(result=response)