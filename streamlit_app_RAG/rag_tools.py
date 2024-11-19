"""Define the Retrievers you can select from"""

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex

from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

from llama_index.core import PromptTemplate

from typing import Dict, Any

def get_retriever(index: VectorStoreIndex, retriever_type: str = "vector", retriever_config: Dict[str, Any] = None):
    
    print(f"Running {retriever_type} Retriever...")
    
    if retriever_type == "vector":
        return index.as_retriever(
            similarity_threshold=retriever_config["similarity_threshold"],
            similarity_top_k=retriever_config["similarity_top_k"],
            )
    elif retriever_type == "query_fusion":
        
        RETRIEVER_SIMILARITY_TOP_K = 20
        FUSION_SIMILARITY_TOP_K = 10
        
        vector_retriever = index.as_retriever(similarity_top_k=RETRIEVER_SIMILARITY_TOP_K)
        bm25_retriever = BM25Retriever.from_defaults(nodes=index.vector_store.get_nodes(), similarity_top_k=RETRIEVER_SIMILARITY_TOP_K)
        
        return QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=FUSION_SIMILARITY_TOP_K,
            num_queries=3,  # Set to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            llm=Settings.llm,
        )
    else:
        raise ValueError(f"Invalid retriever type: {retriever_type}")
    
def get_post_retrieval_transform(post_retrieval_transform_type: str = "similarity"):
    
    print(f"Running {post_retrieval_transform_type} Post-Retrieval Transform...")
    
    if post_retrieval_transform_type == "similarity":
        return SimilarityPostprocessor(
            similarity_cutoff=0.6
            )
    elif post_retrieval_transform_type == "rerank":
        return LLMRerank(
            choice_batch_size=10, 
            top_n=10, 
            llm=Settings.llm,
        )
    elif post_retrieval_transform_type == "sentence_embedding":
        return SentenceEmbeddingOptimizer(
            embed_model=Settings.embed_model,
            #percentile_cutoff=0.5,
            context_before=1,
            context_after=1,
            threshold_cutoff=0.4,
        )
    else:
        raise ValueError(f"Invalid post-retrieval transform type: {post_retrieval_transform_type}")
    
def get_synthesizer(response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE, context_prompt: PromptTemplate = None):
    
    print(f"Running {response_mode} Response Synthesizer...")
    
    return get_response_synthesizer(
            response_mode=response_mode,
            llm=Settings.llm,
            streaming=True,
            verbose=True,
            summary_template=context_prompt,
            )
