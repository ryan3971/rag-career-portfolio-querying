
# Career Portfolio RAG Assistant

A Retrieval-Augmented Generation (RAG) system that enables natural language querying of career portfolio data stored in Notion databases. The system uses LlamaIndex and OpenAI's embedding/LLM services to provide intelligent responses about work experience, projects, and skills.

## Overview

This system consists of two main components:

1. **ETL Pipeline**: Extracts data from Notion databases, processes it into embeddings, and stores them in a vector database

2. **Query Interface**: A Streamlit web application that enables natural language interaction with the portfolio data

## Key Features
- Intelligent natural language querying of portfolio data
- Dual-index architecture (text and keywords) for improved retrieval
- Real-time response streaming
- Debug panel for transparency into the RAG process
- Sample queries to demonstrate capabilities
- Configurable through environment variables


## Project Structure
### Core Files

- `notion_data_etl.ipynb`: Jupyter notebook for extracting and processing Notion data
  - Handles authentication and database connections
  - Processes documents into text and keyword nodes
  - Creates vector indices for efficient retrieval


- `streamlit_app_rag.py`: Main web application interface
  - Implements the RAG Assistant UI
  - Manages chat history and debug output
  - Handles real-time response streaming


- `prompts.py`: Contains system prompts for:
  - Context setting for the LLM
  - Keyword extraction
  
  
### Key Components
#### NotionProcessor Class

A comprehensive data processing class that:

- Extracts data from Notion databases
- Handles nested content structures
- Processes text and metadata
- Supports multiple extraction modes (header, whole, granular)


#### RAGApp Class

The main application class that:

- Manages the Streamlit interface
- Handles chat interactions
- Provides debugging capabilities
- Maintains session state

## Setup Requirements

### Environment Variables
```
NOTION_TOKEN=your_notion_api_token
NOTION_PROJECTS_DATABASE_ID=notion_database_id_for_projects
NOTION_EXPERIENCE_DATABASE_ID=notion_database_id_for_experiences
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Dependencies

- Python 3.10+
- LlamaIndex
- OpenAI
- Streamlit
- Qdrant
- Notion API Client


## Usage

1. Set up the environment variables
2. Run the ETL notebook to process Notion data
3. Run the Streamlit app to interact with the data

```
streamlit run streamlit_app_rag.py
``` 

## Architecture
The system uses a dual-index architecture:

1. **Text Index**: Stores full content for detailed retrieval
2. **Keyword Index**: Stores extracted keywords for improved semantic matching

Queries are processed through both indices to provide comprehensive and accurate responses.


## Debug Features
The system includes a comprehensive debug panel that shows:
- Query processing steps
- Retrieval process details
- Response generation
- Any errors or warnings
