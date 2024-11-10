import os
import sys
from getpass import getpass

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import Accumulate
from llama_index.core.postprocessor.llm_rerank import LLMRerank

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


#     nodes: list[NodeWithScore]

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core import PromptTemplate
from prompts import QUERY_GEN_PROMPT
QUERY_GEN_PROMPT_TEMPLATE = PromptTemplate(QUERY_GEN_PROMPT)


class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None
        
        # define the retrievers
        vector_retriever = index.as_retriever(similarity_top_k=5)
        bm25_retriever = BM25Retriever.from_defaults(nodes=index.vector_store.get_nodes(), similarity_top_k=5)
        
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=3,
            
            num_queries=3,  # Set to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            llm=llm_openai,
            query_gen_prompt=QUERY_GEN_PROMPT_TEMPLATE
        )

        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)
    
    @step
    async def post_retrieval_transform(self, ctx: Context, ev: RetrieverEvent) -> RetrieverEvent:
        """Transform the retrieved node before synthesizing"""
        
        nodes = ev.nodes
        
        

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        
        # exclude metadata from being used in the synthesizer
        nodes = [Node(text=n.node.text) for n in ev.nodes]
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=llm_openai,
            streaming=True,
            verbose=True,
            
            )
        
        query = await ctx.get("query", default=None)

        response = await response_synthesizer.asynthesize(query, nodes=nodes)
        return StopEvent(result=response)





if __name__ == "__main__":
    # Define environment variables
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

    CO_API_KEY = os.environ['CO_API_KEY'] or getpass("Enter your Cohere API key: ")

    QDRANT_URL = os.environ['QDRANT_URL']
    QDRANT_API_KEY = os.environ['QDRANT_API_KEY']

    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    # set up observability with Llamatrace
        
    # Add Phoenix
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )

    # Add them to the tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

    # Instrument the application
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
