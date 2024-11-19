import streamlit as st
from llama_index.core.memory import ChatMemoryBuffer
import os
import sys
from typing import Optional
from io import StringIO
from contextlib import contextmanager
import queue
import threading
from datetime import datetime

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

from typing import Dict, Any

from typing import List

page_style = """
<style>
.main > div {
    padding-top: 1rem;
}
[data-testid="column"] {
    height: calc(100vh - 100px);
    overflow-y: auto !important;
    padding: 1rem;
}
[data-testid="column"] > div {
    height: 100%;
}
.stChatMessage {
    overflow-wrap: break-word;
}
</style>
"""

class RAGConfig:
    """Configuration options for the RAG pipeline"""

    RETRIEVER_OPTIONS = {
        "Similarity Search": "vector",
        "Query Fusion": "query_fusion",
    }
    
    POSTPROCESSOR_OPTIONS = {
        "None": None,
        "Similarity": "similarity",
        "Rerank": "rerank",
        "Sentence Embedding": "sentence_embedding"
    }
    
    RESPONSE_MODE_OPTIONS = {
        "Compact": ResponseMode.COMPACT,
        "Tree Summarize": ResponseMode.TREE_SUMMARIZE,
        "Refine": ResponseMode.REFINE,
        "Simple Summarize": ResponseMode.SIMPLE_SUMMARIZE,
        "Accumulate": ResponseMode.ACCUMULATE
    }
    
class DebugMessage:
    """Structure for debug messages with metadata"""
    def __init__(self, query: str):
        self.query = query
        self.timestamp = datetime.now()
        self.messages: List[str] = []
        self.expanded = False
    
    def add_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.messages.append(f"[{timestamp}] {message}")

    def get_formatted_messages(self) -> str:
        return "\n".join(self.messages)
    
class OutputCapture:
    """Capture and redirect stdout/stderr for debug panel"""
    def __init__(self, debug_message: DebugMessage):
        self.debug_message = debug_message

    def write(self, text):
        if text.strip():  # Only process non-empty strings
            self.debug_message.add_message(text.strip())

    def flush(self):
        pass
    
class RAGApp:
    """
    RAG Application with database connection check and simplified pipeline processing
    """
    
    def __init__(self, pipeline, index_text, index_keywords):
        """
        Initialize the app
        
        Args:
            pipeline: Pre-configured query processing pipeline
            db_connection: Database connection object to check connectivity
        """
        self.pipeline = pipeline
        self.index_text = index_text
        self.index_keywords = index_keywords
        self.debug_queue = queue.Queue()
        self._initialize_session_state()

    def _initialize_session_state(self):
        # set debug mode to true by default
        st.session_state.show_debug = True
        st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
        
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "OPENAI_API_KEY" not in st.session_state:
            st.session_state.OPENAI_API_KEY = None
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []
        if "current_retriever" not in st.session_state:
            st.session_state.current_retriever = "Similarity Search"
        if "current_postprocessor" not in st.session_state:
            st.session_state.current_postprocessor = "Rerank"
        if "current_response_mode" not in st.session_state:
            st.session_state.current_response_mode = "Compact"
        if "retriever_config" not in st.session_state:
            st.session_state.retriever_config = {}
            
    @contextmanager
    def capture_output(self, debug_message: DebugMessage):
        """Context manager to capture stdout/stderr"""
        output_capture = OutputCapture(debug_message)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = output_capture, output_capture
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            
    def update_debug_panel(self, placeholder):
        """Update the debug panel with new messages"""
        while True:
            try:
                message = self.debug_queue.get_nowait()
                st.session_state.debug_messages.append(message)
                # Keep only the last 100 messages
                if len(st.session_state.debug_messages) > 100:
                    st.session_state.debug_messages.pop(0)
                # Update the debug panel
                debug_text = "\n".join(st.session_state.debug_messages)
                placeholder.code(debug_text, language="")
            except queue.Empty:
                break

    def check_openai_key(self) -> bool:
        """Verify OpenAI key is set"""
        return bool(st.session_state.get('OPENAI_API_KEY', ''))

    def check_database_connection(self) -> bool:
        """
        Check if database connection is active
        """
        try:
            # Attempt to verify database connection
            is_connected = True if self.index_text and self.index_keywords else False
            st.session_state.db_connected = is_connected
            return is_connected
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            st.session_state.db_connected = False
            return False

    def init_sidebar(self):
        """Initialize the sidebar with all configurations"""
        with st.sidebar:
            st.header("Configuration")
            
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key to use the application",
                key="openai_key_input"
            )
            if api_key:
                st.session_state.OPENAI_API_KEY = api_key
                os.environ['OPENAI_API_KEY'] = api_key

            # Database connection status
            st.subheader("System Status")
            if st.session_state.db_connected:
                st.success("✓ Connected to database")
            else:
                st.error("✗ Database disconnected")
                if st.button("Retry Connection"):
                    self.check_database_connection()
                    
            # Debug panel toggle
            st.session_state.show_debug = st.toggle(
                "Show Debug Panel",
                value=st.session_state.show_debug,
                help="Show background processing information"
            )

            # RAG Pipeline Configuration
            st.subheader("Pipeline Configuration")

            # Retriever Selection
            st.session_state.current_retriever = st.selectbox(
                "Select Retriever",
                options=list(RAGConfig.RETRIEVER_OPTIONS.keys()),
                index=list(RAGConfig.RETRIEVER_OPTIONS.keys()).index(st.session_state.current_retriever)
            )

            # Retriever-specific configuration
            with st.expander("Retriever Configuration"):
                if st.session_state.current_retriever == "Similarity Search":
                    st.session_state.retriever_config["similarity_top_k"] = st.slider(
                        "Top K Results",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.retriever_config.get("similarity_top_k", 3)
                    )
                    st.session_state.retriever_config["similarity_cutoff"] = st.slider(
                        "Similarity Cutoff",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.retriever_config.get("similarity_cutoff", 0.7)
                    )

            # Postprocessor Selection
            st.session_state.current_postprocessor = st.selectbox(
                "Select Postprocessor",
                options=list(RAGConfig.POSTPROCESSOR_OPTIONS.keys()),
                index=list(RAGConfig.POSTPROCESSOR_OPTIONS.keys()).index(st.session_state.current_postprocessor)
            )

            # Response Mode Selection
            st.session_state.current_response_mode = st.selectbox(
                "Select Response Mode",
                options=list(RAGConfig.RESPONSE_MODE_OPTIONS.keys()),
                index=list(RAGConfig.RESPONSE_MODE_OPTIONS.keys()).index(st.session_state.current_response_mode)
            )


            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.chat_memory.reset()
                st.session_state.debug_messages = []
                st.success("Chat history cleared!")

            # System information
            st.markdown("---")
            st.markdown("""
                ### Usage Tips
                - Ensure database connection is active
                - Be specific in your questions
                - Clear chat history for new sessions
            """)

    def get_current_pipeline_config(self) -> Dict[str, Any]:
        """Get the current pipeline configuration based on sidebar selections"""
        config = {
            "retriever": RAGConfig.RETRIEVER_OPTIONS[st.session_state.current_retriever],
            "retriever_config": st.session_state.retriever_config,
            "postprocessor": RAGConfig.POSTPROCESSOR_OPTIONS[st.session_state.current_postprocessor],
            "response_mode": RAGConfig.RESPONSE_MODE_OPTIONS[st.session_state.current_response_mode]
        }
        return config

    async def process_query_with_streaming(self, query: str, response_placeholder, debug_message: DebugMessage) -> str:
        """
        Process a query with streaming response and debug output
        
        Args:
            query: User's input query
            response_placeholder: Streamlit placeholder for response
            debug_message: DebugMessage object to store debug output
        """
        try:
            if not st.session_state.db_connected:
                return "Sorry, I cannot process your query as the database connection is not active."

            # Initialize empty response
            full_response = ""
            
            debug_message.add_message(f"Pipeline Configuration:")
            debug_message.add_message(f"- Retriever: {st.session_state.current_retriever}")
            debug_message.add_message(f"- Retriever Config: {st.session_state.retriever_config}")
            debug_message.add_message(f"- Postprocessor: {st.session_state.current_postprocessor}")
            debug_message.add_message(f"- Response Mode: {st.session_state.current_response_mode}")

            # Capture all stdout/stderr during processing
            with self.capture_output(debug_message):
                print(f"Processing query: {query}")
                print("Initializing pipeline...")
                
                # Get the current pipeline configuration
                pipeline_config = self.get_current_pipeline_config()
                result = await self.pipeline.run(
                    query=query, 
                    index_text=self.index_text, 
                    index_keywords=self.index_keywords,
                    config=pipeline_config
                )
                
                # Stream the response
                print("Streaming response...")
                async for chunk in result.async_response_gen():
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                    
                print("Response complete")
                print(f"Final response length: {len(full_response)} characters")
            
            # Final response update
            response_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            debug_message.add_message(error_msg)
            print(error_msg)  # This will be captured in debug output
            st.error(error_msg)
            return "Sorry, I encountered an error while processing your query."

    def render_debug_panel(self):
        """Render the debug panel with expandable messages"""
        st.markdown("<h3>Debug Output</h3>", unsafe_allow_html=True)
        
        for debug_msg in reversed(st.session_state.debug_messages):  # Show newest first
            with st.expander(
                f"Query: {debug_msg.query[:50]}... ({debug_msg.timestamp.strftime('%H:%M:%S')})",
                expanded=False
            ):
                # Display all captured output
                st.code(debug_msg.get_formatted_messages(), language="")

    async def render_chat_area(self):
        # Check database connection
        if not self.check_database_connection():
            st.warning("Cannot process queries: Database connection is not active.")
            return
        
        # Check for API key
        if not self.check_openai_key():
            st.warning("Please enter your OpenAI API key in the sidebar to continue.")
            return
        
        # Create container for chat
        chat_container = st.container()
        
        # Create container for input at the bottom
        input_container = st.container()
        
        # Use the chat container for messages
        with chat_container:
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Use the input container for the chat input
        with input_container:
            st.markdown('<div class="chat-input">', unsafe_allow_html=True)
            # Chat input
            if prompt := st.chat_input(
                "What would you like to know?",
                disabled=not st.session_state.db_connected
            ):
                # Create debug message object for this query
                debug_msg = DebugMessage(prompt)
                st.session_state.debug_messages.append(debug_msg)
                
                # Display user message (in chat container)
                with chat_container:
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Create placeholder for assistant response
                    assistant_response = st.chat_message("assistant")
                    response_placeholder = assistant_response.empty()

                    # Process query
                    full_response = await self.process_query_with_streaming(
                        prompt,
                        response_placeholder,
                        debug_msg
                    )
                    
                    # Store the complete response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
            st.markdown('</div>', unsafe_allow_html=True)

    async def run(self):
        """Run the Streamlit application"""
        st.title("💡 RAG Assistant")
        
        # Initialize sidebar
        self.init_sidebar()
                
        # Create two-column layout
        chat_col, debug_col = st.columns(2)
        
        #st.markdown(page_style, unsafe_allow_html=True)
        
        # Render chat area in left column
        with chat_col.container():
            await self.render_chat_area()
        
        # Render debug panel in right column
        with debug_col.container():
            self.render_debug_panel()



from rag_pipeline import RAGWorkflow
from rag_setup import index_database_connection, set_up_llm, set_up_embeddings

async def main():
    """Main entry point for the application"""
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # initialize the LLM and embeddings
    llm = set_up_llm()
    embed_model = set_up_embeddings()

    # Initialize your pipeline and database connection here
    pipeline = RAGWorkflow(llm, embed_model)
    index_text, index_keywords = index_database_connection()
    
    # Initialize and run app
    app = RAGApp(pipeline, index_text, index_keywords)
    await app.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())