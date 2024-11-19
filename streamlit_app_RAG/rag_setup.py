import os
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex

from prompts import LLM_CONTEXT_PROMPT

COLLECTION_TEXT = "Notion_vector_store_text"
COLLECTION_KEYWORD = "Notion_vector_store_keywords"

def set_up_llm():
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Set up the LLM
    llm_openai = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.5,
        api_key=OPENAI_API_KEY,
        system_prompt=LLM_CONTEXT_PROMPT
    )

    Settings.llm = llm_openai
    
    #return llm_openai
    
def set_up_embeddings():
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Set up the Embeddings
    embed_model_openai = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY
        )
    Settings.embed_model = embed_model_openai
    
    #return embed_model_openai

def index_database_connection():
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # set up the vector store client
    client = QdrantClient(
        location=QDRANT_URL, 
        api_key=QDRANT_API_KEY
    )
    # set up the async client
    aclient = AsyncQdrantClient(
        location=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    # set up the vector store
    
    vector_store_text = QdrantVectorStore(
        client=client, 
        aclient=aclient, 
        collection_name=COLLECTION_TEXT,
    )
    
    vector_store_keywords = QdrantVectorStore(
        client=client, 
        aclient=aclient, 
        collection_name=COLLECTION_KEYWORD,
    )
    
    storage_context_text = StorageContext.from_defaults(
        vector_store=vector_store_text
        )
    
    # storage context for caching (I think)
    storage_context_keywords = StorageContext.from_defaults(
        vector_store=vector_store_keywords
        )
    
    index_text = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_text,
        storage_context=storage_context_text,
        show_progress=True
        )

    index_keywords = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_keywords,
        storage_context=storage_context_keywords,
        show_progress=True
        )
    
    # index_keyword_regex = KeywordTableIndex(
    #     vector_store=vector_store_text,
    #     text_key="text",
    #     regex_key="regex"
    # )
    
    return index_text, index_keywords#, index_keyword_regex
