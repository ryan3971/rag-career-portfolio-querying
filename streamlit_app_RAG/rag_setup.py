import os
from typing import Tuple
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings

from prompts import LLM_CONTEXT_PROMPT

# Collection names as constants
COLLECTION_TEXT = os.getenv("COLLECTION_TEXT")
COLLECTION_KEYWORD = os.getenv("COLLECTION_KEYWORD")

# Cache API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def set_up_llm() -> None:
    """Initialize and configure the OpenAI LLM."""
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=OPENAI_API_KEY,
        system_prompt=LLM_CONTEXT_PROMPT
    )
        
def set_up_embeddings() -> None:
    """Initialize and configure the OpenAI embeddings model."""
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY
    )
    
def index_database_connection() -> Tuple[VectorStoreIndex, VectorStoreIndex]:
    """Set up connection to Qdrant and create vector store indices."""
    # Initialize Qdrant clients
    client = QdrantClient(
        location=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    aclient = AsyncQdrantClient(
        location=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Create vector stores for text and keywords
    stores = {
        COLLECTION_TEXT: QdrantVectorStore(client=client, aclient=aclient, collection_name=COLLECTION_TEXT),
        COLLECTION_KEYWORD: QdrantVectorStore(client=client, aclient=aclient, collection_name=COLLECTION_KEYWORD)
    }
    
    # Create indices from vector stores
    indices = {}
    for name, store in stores.items():
        storage_context = StorageContext.from_defaults(vector_store=store)
        indices[name] = VectorStoreIndex.from_vector_store(
            vector_store=store,
            storage_context=storage_context,
            show_progress=True
        )
    
    # Create retriever for keywords index
    indices[COLLECTION_KEYWORD].as_retriever()
    
    return indices[COLLECTION_TEXT], indices[COLLECTION_KEYWORD]
