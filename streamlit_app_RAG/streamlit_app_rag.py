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

class OutputCapture:
    """Capture and redirect stdout/stderr for debug panel"""
    def __init__(self, queue: queue.Queue):
        self.queue = queue

    def write(self, text):
        if text.strip():  # Only process non-empty strings
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.queue.put(f"[{timestamp}] {text}")

    def flush(self):
        pass

class RAGApp:
    """
    RAG Application with database connection check and simplified pipeline processing
    """
    def __init__(self, pipeline, index):
        """
        Initialize the app
        
        Args:
            pipeline: Pre-configured query processing pipeline
            db_connection: Database connection object to check connectivity
        """
        self.pipeline = pipeline
        self.index = index
        self.debug_queue = queue.Queue()
        self._initialize_session_state()
        self._setup_page()

    def _initialize_session_state(self):
        
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "OPENAI_API_KEY" not in st.session_state:
            try:
                st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            except:
                st.session_state.OPENAI_API_KEY = None
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False
        if "show_debug" not in st.session_state:
            st.session_state.show_debug = False
        if "debug_messages" not in st.session_state:
            st.session_state.debug_messages = []
            

    def _setup_page(self):
        st.markdown("""
            <style>
            .debug-panel {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: monospace;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                max-height: 400px;
                overflow-y: auto;
            }
            .timestamp {
                color: #888;
                margin-right: 10px;
            }
            </style>
        """, unsafe_allow_html=True)
        
    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout/stderr"""
        output_capture = OutputCapture(self.debug_queue)
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
            is_connected = True if self.index else False
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

    async def process_query_with_streaming(self, query: str, response_placeholder, debug_placeholder) -> str:
        """
        Process a query and stream the response
        
        Args:
            query: User's input query
            placeholder: Streamlit placeholder for streaming output
            
        Returns:
            str: Complete response after streaming
        """
        try:
            if not st.session_state.db_connected:
                return "Sorry, I cannot process your query as the database connection is not active."

            # Initialize empty response
            full_response = ""
            # Capture and display debug output during processing
            with self.capture_output():
                print(f"Processing query: {query}")
                print("Initializing pipeline...")
                
                result = await self.pipeline.run(query=query, index=self.index)
                
                print("Streaming response...")
                async for chunk in result.async_response_gen():
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                    # Update debug panel if it exists
                    if debug_placeholder is not None:
                        self.update_debug_panel(debug_placeholder)
                print("Response complete")
            
            # Final updates
            response_placeholder.markdown(full_response)
            if debug_placeholder is not None:
                self.update_debug_panel(debug_placeholder)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            if debug_placeholder is not None:
                self.update_debug_panel(debug_placeholder)
            st.error(error_msg)
            return "Sorry, I encountered an error while processing your query."

    async def render_chat_area(self):
        # Check database connection
        if not self.check_database_connection():
            st.warning("Cannot process queries: Database connection is not active.")
            return
        
        # Check for API key
        if not self.check_openai_key():
            st.warning("Please enter your OpenAI API key in the sidebar to continue.")
            return

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create debug placeholder if debug mode is enabled
        debug_placeholder = st.empty() if st.session_state.show_debug else None

        # Chat input
        if prompt := st.chat_input(
            "What would you like to know?",
            disabled=not st.session_state.db_connected
        ):
        
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Create a placeholder for the assistant's response
            assistant_response = st.chat_message("assistant")
            response_placeholder = assistant_response.empty()
                    
            # Process query with streaming response
            full_response = await self.process_query_with_streaming(
                prompt,
                response_placeholder,
                debug_placeholder
            )
            
            # Store the complete response in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

    async def run(self):
        """Run the Streamlit application"""
        st.title("💡 RAG Assistant")
        
        # Initialize sidebar
        self.init_sidebar()
        
        # Create layout
        if st.session_state.show_debug:
            # Create columns
            cols = st.columns([2, 1])
            
            # Main chat area
            with cols[0]:
                await self.render_chat_area()
            
            # Debug panel
            with cols[1]:
                st.subheader("Debug Output")
                debug_placeholder = st.empty()
                
        else:
            # Just render chat area without debug panel
            await self.render_chat_area()
            debug_placeholder = None



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
    index = index_database_connection()
    
    # Initialize and run app
    app = RAGApp(pipeline, index)
    await app.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())