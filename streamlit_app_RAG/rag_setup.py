from llama_index.llms.cohere import Cohere
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import StorageContext
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex


from llama_index.core.settings import Settings

import os

COLLECTION_NAME = "Notion_vector_store"

def set_up_llm():
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Set up the LLM
    llm_openai = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.5,
        api_key=OPENAI_API_KEY,
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
    vector_store = QdrantVectorStore(
        client=client, 
        aclient=aclient, 
            collection_name=COLLECTION_NAME,
    )
    
    # storage context for caching (I think)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
        )

    # Create index from vector store with storage context
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,  # Add storage context
    #    embed_model=embed_model_openai,
        show_progress=True  # Optional: helps track progress
    )
    
    return index
