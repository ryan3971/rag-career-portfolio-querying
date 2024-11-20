import streamlit as st
import os
import sys
from contextlib import contextmanager
import queue
from datetime import datetime


# Sample queries for the dropdown
SAMPLE_QUERIES = {
    "Select a sample query...": "",
    "What projects have you worked on?": "What projects have you worked on and what technologies did you use?",
    "Tell me about your skills": "What are your main technical skills and expertise?",
    "Work experience": "Can you describe your work experience and key achievements?",
    "Education background": "What is your educational background and relevant certifications?",
    "Leadership experience": "Can you tell me about your leadership experience and team management skills?"
}

page_style = """
<style>
.main {
    padding: 0 !important;
}

.debug-panel {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-top: 2px solid #ddd;
    padding: 1rem;
    transform: translateY(100%);
    transition: transform 0.3s ease-out;
    z-index: 1000;
    height: 50vh;
    overflow-y: auto;
}

.debug-panel.open {
    transform: translateY(0);
}

.sample-query-container {
    margin-bottom: 2rem;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
}

.query-count {
    color: #666;
    font-size: 0.9em;
    margin-top: 0.5rem;
}
</style>
""" 
class DebugMessage:
    """Structured debug message with organized sections"""
    def __init__(self, query: str):
        self.query = query
        self.timestamp = datetime.now()
        self.sections = {
            "query_info": [],      # Basic query information
            "retrieval": [],       # Retrieval-related information
            "processing": [],      # Processing steps
            "response": [],        # Response generation
            "errors": []           # Any errors that occurred
        }

    def add_to_section(self, section: str, message: str):
        """Add a message to a specific section"""
        if section in self.sections:
            self.sections[section].append(message.strip())

    def get_formatted_section(self, section: str, title: str) -> str:
        """Format a section's messages with a title"""
        if not self.sections[section]:
            return ""
        
        messages = self.sections[section]
        return f"{title}:\n" + "\n".join(f"  {msg}" for msg in messages)

    def get_formatted_messages(self) -> str:
        """Get all sections formatted nicely"""
        sections_text = []
        
        # Query Information
        if self.sections["query_info"]:
            sections_text.append(self.get_formatted_section("query_info", "Query Information"))
        
        # Retrieval Information
        if self.sections["retrieval"]:
            sections_text.append(self.get_formatted_section("retrieval", "Retrieval Process"))
        
        # Processing Steps
        if self.sections["processing"]:
            sections_text.append(self.get_formatted_section("processing", "Processing Steps"))
        
        # Response Generation
        if self.sections["response"]:
            sections_text.append(self.get_formatted_section("response", "Response Generation"))
        
        # Errors (only if present)
        if self.sections["errors"]:
            sections_text.append(self.get_formatted_section("errors", "⚠️ Errors"))

        return "\n\n".join(section for section in sections_text if section)

class OutputCapture:
    """Capture and redirect stdout/stderr with section routing"""
    def __init__(self, debug_message: DebugMessage):
        self.debug_message = debug_message

    def write(self, text):
        """Capture and route output to debug message sections"""
        if text.strip():
            # Check for section prefix in square brackets
            text = text.strip()
            if text.startswith('['):
                try:
                    section = text[1:text.index(']')]
                    message = text[text.index(']')+1:].strip()
                    self.debug_message.add_to_section(section, message)
                except ValueError:
                    # If message isn't properly formatted, add to processing
                    self.debug_message.add_to_section('processing', text)
            else:
                # Default to processing section if no prefix
                self.debug_message.add_to_section('processing', text)

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
        """Initialize all session state variables with default values"""
        defaults = {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
            'using_default_key': True,
            'messages': [],
            'db_connected': False,
            'debug_messages': [],
            'previous_query': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @contextmanager
    def capture_output(self, debug_message: DebugMessage):
        """Context manager to capture stdout/stderr"""
        output_capture = OutputCapture(debug_message)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = output_capture, output_capture
        try:
            yield output_capture
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
            # Database connection status
            st.subheader("Database Status")
            if st.session_state.db_connected:
                st.success("✓ Connected to database")
            else:
                st.error("✗ Database disconnected")
                if st.button("Retry Connection"):
                    self.check_database_connection()

            # Tips Section
            st.markdown("---")
            st.markdown("""
                ### Tips
                - Select from sample queries or type your own
                - Use specific questions for better results
                - Check the debug panel to understand the process
                - First 5 queries are free
                - Provide your API key for unlimited queries
            """)
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.debug_messages = []
                st.success("Chat history cleared!")

    async def process_query_with_streaming(self, query: str, response_placeholder, debug_message: DebugMessage) -> str:
        """Process a query with streaming response and debug output"""
        try:
            if not st.session_state.db_connected:
                debug_message.add_to_section("errors", "Database connection is not active")
                return "Sorry, I cannot process your query as the database connection is not active."

            full_response = ""
            
            # Capture and display debug output during processing
            with self.capture_output(debug_message):
                # Query Information
                print(f"[query_info] Input query: {query}")
                print(f"[query_info] Query length: {len(query)} characters")
                
                # Processing
                print("[processing] Starting post-retrieval processing")
                
                # Process through pipeline
                result = await self.pipeline.run(query=query, index_text=self.index_text, index_keywords=self.index_keywords)
                
                # Response Generation
                print("[response] Starting response generation")
                
                async for chunk in result.async_response_gen():
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                print(f"[response] Response length: {len(full_response)} characters")
                print("[response] Response generation complete")
            
            response_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            debug_message.add_to_section("errors", error_msg)
            st.error(error_msg)
            return "Sorry, I encountered an error while processing your query."

    def render_debug_panel(self):
        """Render the debug panel with organized sections"""
        with st.container():
            st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
            
            # Simplified debug panel header
            st.markdown("""
                <div style='margin-bottom: 1rem;'>
                    <h3 style='margin: 0;'>Debug Output</h3>
                    <p style='color: #666; margin: 0;'>Showing process details and diagnostics</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display debug messages in reverse chronological order
            for debug_msg in reversed(st.session_state.debug_messages):
                self._render_debug_message(debug_msg)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _render_debug_message(self, debug_msg):
        """Helper method to render individual debug messages"""
        with st.expander(f"Query: {debug_msg.query[:50]}...", expanded=False):
            # Show errors first if any exist
            if debug_msg.sections["errors"]:
                st.error(debug_msg.get_formatted_section("errors", "⚠️ Errors"))
            
            # Create two-column layout for debug info
            col1, col2 = st.columns(2)
            
            # Left column: Query info and processing steps
            with col1:
                self._render_debug_section(debug_msg, "query_info", "Query Information")
                self._render_debug_section(debug_msg, "processing", "Processing Steps")
            
            # Right column: Retrieval and response info
            with col2:
                self._render_debug_section(debug_msg, "retrieval", "Retrieval Process")
                self._render_debug_section(debug_msg, "response", "Response Generation")

    def _render_debug_section(self, debug_msg, section_name: str, title: str):
        """Helper method to render a debug section"""
        if debug_msg.sections[section_name]:
            st.markdown(f"**{title}**")
            st.code("\n".join(debug_msg.sections[section_name]), language="")

    async def render_chat_area(self):
        """Render the chat area with query selection and input"""
        # Check database connection
        if not self.check_database_connection():
            st.warning("Cannot process queries: Database connection is not active.")
            return
        
        # Initialize the session state for query selection
        if 'previous_query' not in st.session_state:
            st.session_state.previous_query = None
        
        selected_query = st.selectbox(
            "Try a sample query or type your own below:",
            options=list(SAMPLE_QUERIES.keys()),
            key='query_selector'
        )
        
        # Create containers for chat and input
        chat_container = st.container()
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
            user_input  = st.chat_input(
                "What would you like to know?",
                disabled=not st.session_state.db_connected
            )
            
            # Simplified prompt handling
            prompt = None
            if user_input:
                prompt = user_input
                st.session_state.previous_query = None
            elif selected_query != "Select a sample query..." and selected_query != st.session_state.previous_query:
                prompt = SAMPLE_QUERIES[selected_query]
                st.session_state.previous_query = selected_query

            if prompt:
                await self._process_chat_message(prompt, chat_container)

    async def _process_chat_message(self, prompt: str, chat_container):
        """Helper method to process and display chat messages"""
        debug_msg = DebugMessage(prompt)
        st.session_state.debug_messages.append(debug_msg)
        
        with chat_container:
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Process and display assistant response
            assistant_response = st.chat_message("assistant")
            response_placeholder = assistant_response.empty()

            full_response = await self.process_query_with_streaming(
                prompt,
                response_placeholder,
                debug_msg
            )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

    async def run(self):
        """Run the Streamlit application"""
        st.markdown("# 📚 **RAG Assistant for Career Portfolio Querying**")
        
        # Initialize sidebar
        self.init_sidebar()        
        
        # Render chat area in left column
        await self.render_chat_area()
        
        # Render debug panel in right column
        self.render_debug_panel()

from rag_pipeline import RAGWorkflow
from rag_setup import index_database_connection, set_up_llm, set_up_embeddings

async def main():
    """Main entry point for the application"""
    st.set_page_config(
        page_title="Career Portfolio RAG Assistant",
        page_icon="📚",
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