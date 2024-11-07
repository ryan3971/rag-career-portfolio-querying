import streamlit as st
import openai
import random
import time
import json
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Multi-Model Chat Room",
    page_icon="🤖",
    layout="wide"
)

# Participant class
class ChatParticipant:
    def __init__(self, name, model, personality, color):
        self.name = name
        self.model = model
        self.personality = personality
        self.color = color

    def to_dict(self):
        return {
            "name": self.name,
            "model": self.model,
            "personality": self.personality,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            model=data["model"],
            personality=data["personality"],
            color=data["color"]
        )

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config.json exists in the same directory.")
        return None

# Initialize configuration
if "config" not in st.session_state:
    st.session_state.config = load_config()

# Initialize session states
if "chat_participants" not in st.session_state:
    st.session_state.chat_participants = []
    if st.session_state.config:
        for participant_data in st.session_state.config["participants"]:
            participant = ChatParticipant.from_dict(participant_data)
            st.session_state.chat_participants.append(participant)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_speaker" not in st.session_state:
    st.session_state.current_speaker = None

def generate_response(participant, messages):
    """Generate response using OpenAI API"""
    config = st.session_state.config["chat_settings"]
    try:
        system_prompt = f"""You are {participant.name}. {participant.personality}
        You are participating in a group conversation. Keep your responses concise and engaging.
        Respond in a way that naturally flows with the conversation and occasionally interact with other participants.
        """
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        # Create empty placeholder for streaming
        message_placeholder = st.empty()
        full_response = ""
        
        # Create a streaming completion with OpenAI API
        # stream=True enables real-time response streaming
        for response in openai.chat.completions.create(
            model=participant.model,            # The GPT model to use (e.g. gpt-3.5-turbo)
            messages=formatted_messages,        # List of conversation messages
            temperature=config["temperature"],  # Controls randomness (0-1)
            max_tokens=config["max_tokens"],    # Max length of response
            stream=True                        # Enable streaming mode
        ):
            # Get the new piece of text from the streamed response
            content = response.choices[0].delta.content
            if content is not None:
                # Add the new content to build up the full response
                full_response += content
                
                # Update the UI in real-time with the growing response
                # Uses markdown formatting with:
                # - Participant's custom color
                # - Bold name prefix
                # - Cursor symbol (▌) to show it's still typing
                message_placeholder.markdown(f'<span style="color: {participant.color}">'
                                          f'**{participant.name}**: {full_response}▌</span>',
                                          unsafe_allow_html=True)
        
        # Clear the placeholder
        message_placeholder.empty()
        return full_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("Chat Room Settings")
    
    # Load API key from config if available
    if st.session_state.config:
        api_key = st.session_state.config["openai_api_key"]
        if api_key != "your-api-key-here":
            openai.api_key = api_key
        else:
            api_key = st.text_input("OpenAI API Key:", type="password")
            if api_key:
                openai.api_key = api_key
    
    # Display current participants
    st.subheader("Current Participants")
    for participant in st.session_state.chat_participants:
        st.markdown(
            f'<div style="padding: 10px; border-radius: 5px; margin: 5px 0; background-color: {participant.color}20;">'
            f'<strong>{participant.name}</strong><br>'
            f'Model: {participant.model}<br>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_speaker = None

# Main chat interface
st.title("Multi-Model Chat Room 🤖")

# Topic starter
if not st.session_state.messages:
    if st.session_state.config:
        default_topics = st.session_state.config["default_topics"]
        topic = st.selectbox("Select a conversation topic:", 
                           options=default_topics,
                           index=None,
                           placeholder="Choose a topic or write your own below")
        
        custom_topic = st.text_input("Or write your own topic:")
        final_topic = custom_topic if custom_topic else topic
        
        if st.button("Begin Conversation") and final_topic:
            st.session_state.messages.append({
                "role": "system",
                "content": f"The conversation topic is: {final_topic}"
            })
            st.session_state.current_speaker = random.choice(st.session_state.chat_participants)

# Display chat
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"], avatar="🤖"):
            st.markdown(f'<span style="color: {msg.get("color", "#ffffff")}">'
                       f'**{msg.get("name", "System")}**: {msg["content"]}</span>',
                       unsafe_allow_html=True)

# Generate next response
if st.session_state.messages and len(st.session_state.chat_participants) > 1:
    if st.button("Generate Next Response"):
        # Only proceed if there are messages and multiple participants
        
        # Get all participants except current speaker to choose next speaker from
        available_participants = [p for p in st.session_state.chat_participants 
                                if p.name != (st.session_state.current_speaker.name 
                                            if st.session_state.current_speaker else None)]
        # Randomly select next speaker from available participants                                    
        next_speaker = random.choice(available_participants)
        
        # Get the max number of previous messages to include as context
        max_context = st.session_state.config["chat_settings"]["max_context_messages"]
        
        # Format recent messages to provide context for next response
        # Takes last N messages and formats them as:
        # "Speaker Name: Message content" for chat messages
        # Or just content for system messages
        context_messages = [
            {
                "role": "assistant" if msg["role"] == "assistant" else msg["role"],
                "content": f"{msg.get('name', 'System')}: {msg['content']}"
                if msg["role"] != "system" else msg["content"]
            }
            for msg in st.session_state.messages[-max_context:]
        ]
        
        # Generate response from next speaker using context
        response = generate_response(next_speaker, context_messages)
        
        # Add the generated response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "name": next_speaker.name,
            "color": next_speaker.color
        })
        
        # Update current speaker
        st.session_state.current_speaker = next_speaker
        
        # Refresh page to show new message
        st.rerun()

# Export chat
if st.session_state.messages:
    if st.button("Export Conversation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_data = {
            "timestamp": timestamp,
            "participants": [p.to_dict() for p in st.session_state.chat_participants],
            "messages": st.session_state.messages
        }
        st.download_button(
            label="Download Conversation",
            data=json.dumps(export_data, indent=2),
            file_name=f"conversation_{timestamp}.json",
            mime="application/json"
        )