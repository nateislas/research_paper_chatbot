# Research Assistant AI

An advanced, multi-agent document analysis platform built with **LlamaIndex Workflows** and **LlamaCloud Managed RAG**. This system provides sophisticated research capabilities that go beyond simple retrieval, using intelligent planning, parallel execution, and comprehensive synthesis to answer complex questions about research papers and technical documentation.

## Key Features

### Intelligent Research Workflow
- **Multi-Step Planning**: Automatically decomposes complex questions into 2-4 focused research sub-tasks
- **Parallel Execution**: Executes research steps concurrently using 4 worker threads for faster results
- **Synthesis Engine**: Combines findings from multiple research steps into comprehensive, well-structured answers
- **Event-Driven Architecture**: Real-time progress updates via event streaming

### Managed RAG Infrastructure
- **LlamaCloud Integration**: Fully managed vector indexing - no local database setup required
- **LlamaParse Integration**: High-resolution agentic parsing for complex PDFs with tables, charts, and hierarchical structures
- **Automatic Ingestion**: Handles document parsing, embedding, and indexing in the cloud
- **File Management**: Track and filter documents by file ID for targeted queries

### Advanced Query Capabilities
- **Single-Turn Queries**: Fast, stateless question-answering (backward compatible)
- **Multi-Turn Chat**: Conversational mode with memory and context retention
- **File Filtering**: Select specific documents to use as context for queries
- **Citation Tracking**: Automatic extraction of source documents with page numbers and relevance scores

### Premium UI Experience
- **Custom Streamlit Interface**: Modern, clean design with custom CSS styling
- **Real-Time Updates**: Live progress indicators during research execution
- **Citation Display**: Interactive source cards showing file names, page numbers, and extracted text
- **Suggested Questions**: One-click starter prompts for common research tasks

## Architecture

### Workflow Engine

The system uses **LlamaIndex Workflows** for event-driven orchestration:

```
User Question
    ↓
Research Planning (LLM generates 2-4 sub-questions)
    ↓
Parallel Research Steps (4 workers, each queries the index)
    ↓
Synthesis (LLM combines findings)
    ↓
Final Answer + Citations
```

### Core Components

**1. `DocumentChatbotWorkflow`** (`workflow.py`)
- Main workflow class inheriting from `Workflow`
- Handles document upload, query, chat, and file listing
- Implements research planning, execution, and synthesis steps
- Manages LlamaCloud index connections and state

**2. Research Process**
- **Planning Step**: Uses LLM to generate structured research plan (`ResearchPlan` with multiple `ResearchPlanItem`s)
- **Execution Step**: Parallel workers (4) execute each research item, querying the index with filters
- **Synthesis Step**: Collects all results and synthesizes into final answer with citations

**3. Thread-Safe Execution**
- **`WorkflowRunner`**: Dedicated background thread with its own event loop
- **`SyncDocumentChatbot`**: Synchronous wrapper for Streamlit compatibility
- Bridges async workflows with synchronous Streamlit context

**4. UI Layer** (`app.py`)
- Streamlit frontend with custom CSS styling
- Real-time event streaming during research execution
- Document upload and file selection interface
- Citation display with expandable source cards

### Event Flow

The workflow uses custom events for orchestration:

- `StartEvent` → Initial request (upload/query/chat/list_files)
- `ResearchPlanStartEvent` → Triggers research planning
- `ResearchPlanItem` → Individual research step (dispatched in parallel)
- `ResearchPlanItemResult` → Result from each research step
- `ChatbotStopEvent` → Final response with citations

### State Management

- **WorkflowState**: Tracks index ID, conversation IDs, file counts, and chat engines
- **Context Store**: Persistent state across workflow steps
- **Chat Engine Caching**: Reuses chat engines per conversation for multi-turn chats

## Technical Stack

- **Orchestration**: `llama-index-workflows` (Event-driven, async-first)
- **LLM**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Embeddings**: Managed by LlamaCloud (automatic)
- **Vector Store**: LlamaCloud Managed Index (cloud-based)
- **Parsing**: LlamaParse (agentic mode with high-res OCR)
- **Frontend**: Streamlit with custom CSS
- **Async Bridge**: Thread-safe event loop runner for Streamlit compatibility

## Installation

### Prerequisites

- Python 3.10+
- [Google Gemini API Key](https://aistudio.google.com/)
- [LlamaCloud API Key](https://cloud.llamaindex.ai/)

### Setup

1. **Clone and Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   python setup_env.py
   ```
   
   Edit the `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   LLAMA_API_KEY=your_llama_api_key
   LLAMA_CLOUD_PROJECT=Default
   LLAMA_CLOUD_ORG_ID=your_org_id  # Optional
   ```

3. **Launch Application**:
   ```bash
   streamlit run research_paper_chatbot/app.py
   ```
   
   Or use the convenience script:
   ```bash
   ./run.sh
   ```

## Usage

### Web Interface

1. **Upload Documents**: Use the upload button to add PDF, DOCX, PPTX, XLSX, TXT, or MD files
2. **Select Context**: Choose specific documents from the sidebar to filter queries
3. **Ask Questions**: Type questions or use suggested prompts
4. **View Citations**: Expand "Verified Sources" to see source documents with page numbers

### Programmatic API

#### Async API (`DocumentChatbot`)

```python
from research_paper_chatbot.workflow import DocumentChatbot

# Initialize
chatbot = DocumentChatbot(
    google_api_key="your_key",
    llama_api_key="your_key",
    project_name="Default"
)

# Upload documents
await chatbot.upload_document("paper.pdf")
await chatbot.upload_directory("./papers/")

# Single-turn query
response = await chatbot.query("What are the main findings?")

# Multi-turn chat with file filtering
result = await chatbot.chat(
    message="Summarize the methodology",
    conversation_id="conv_123",
    file_ids=["file_id_1", "file_id_2"]
)
print(result["response"])
print(result["citations"])

# List indexed files
files = await chatbot.list_files()
```

#### Sync API (`SyncDocumentChatbot`)

```python
from research_paper_chatbot.workflow import SyncDocumentChatbot

# Initialize (thread-safe for Streamlit)
chatbot = SyncDocumentChatbot(
    google_api_key="your_key",
    llama_api_key="your_key"
)

# Upload
chatbot.upload_document("paper.pdf")

# Query
response = chatbot.query("What is the conclusion?")

# Chat with memory
result = chatbot.chat(
    message="Tell me more about that",
    conversation_id="conv_123"
)
```

#### Event Streaming

For real-time progress updates:

```python
# Start workflow and stream events
handler = await chatbot.run_chat_flow(
    message="Complex research question",
    conversation_id="conv_123"
)

# Stream events as they occur
async for event in handler.stream_events():
    if hasattr(event, "msg"):
        print(f"Progress: {event.msg}")

# Get final result
result = await handler
```

### Command Line Interface

```bash
# Upload a document
python -m research_paper_chatbot.workflow \
    --upload paper.pdf \
    --google-api-key YOUR_KEY \
    --llama-api-key YOUR_KEY

# Upload a directory
python -m research_paper_chatbot.workflow \
    --upload-dir ./papers/ \
    --google-api-key YOUR_KEY \
    --llama-api-key YOUR_KEY

# Query documents
python -m research_paper_chatbot.workflow \
    --query "What are the main findings?" \
    --google-api-key YOUR_KEY \
    --llama-api-key YOUR_KEY
```

## Project Structure

```
research_paper_chatbot/
├── research_paper_chatbot/
│   ├── __init__.py          # Package initialization
│   ├── app.py               # Streamlit UI (796 lines)
│   └── workflow.py          # Workflow engine (1237 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── run.sh                    # Local launch script
├── setup_env.py             # Environment setup utility
└── chatbot.py               # Legacy CLI chatbot (deprecated)
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `LLAMA_API_KEY` | Yes | LlamaCloud API key |
| `LLAMA_CLOUD_PROJECT` | No | Project name (default: "Default") |
| `LLAMA_CLOUD_ORG_ID` | No | Organization ID (if required) |

### Workflow Parameters

- `gemini_model`: LLM model (default: "gemini-2.5-flash")
- `project_name`: LlamaCloud project name (default: "Default")
- `collection_name`: Index/pipeline name (default: "research_papers")
- `timeout`: Workflow timeout in seconds (default: 300)

## UI Features

### Custom Styling
- Modern SaaS-style design with Inter font
- Glassmorphism effects with subtle shadows
- Responsive layout with fluid width
- Custom message bubbles (blue for user, white for assistant)

### Interactive Elements
- Real-time status indicators
- Expandable citation cards
- File selection multiselect
- Suggested question buttons
- Progress updates during research

## How It Works

### Research Planning

When a user asks a complex question, the system:

1. **Analyzes the Question**: LLM determines if the question requires multi-step research
2. **Generates Plan**: Creates 2-4 focused research steps with specific queries
3. **Example**: "Compare the methodologies in these papers"
   - Step 1: "Find methodology descriptions in paper A"
   - Step 2: "Find methodology descriptions in paper B"
   - Step 3: "Identify key differences"

### Parallel Execution

Each research step runs in parallel:

- 4 worker threads execute simultaneously
- Each worker queries the LlamaCloud index independently
- Results include source documents with metadata
- Citations extracted with file names and page numbers

### Synthesis

Final step combines all research findings:

- Aggregates results from all research steps
- Deduplicates citations
- LLM synthesizes into coherent answer
- Returns structured response with citations

### File Filtering

Users can select specific documents for context:

- Metadata filters applied to index queries
- Only selected files used as retrieval context
- Supports multi-file selection
- Filters persist across conversation turns

## Troubleshooting

### Common Issues

**"Google API key is required"**
- Ensure `GOOGLE_API_KEY` is set in environment or `.env` file
- Check that the key is valid and has Gemini API access

**"LlamaCloud index error"**
- Verify `LLAMA_API_KEY` is correct
- Check that project name exists in LlamaCloud
- Ensure organization ID is set if required

**"No documents found"**
- Upload documents first using the upload interface
- Check that file formats are supported (PDF, DOCX, etc.)
- Verify files were successfully indexed (check file list)

**Streamlit async errors**
- The `WorkflowRunner` handles thread-safe execution
- If issues persist, ensure you're using `SyncDocumentChatbot` in Streamlit

## License

This project is open source. See LICENSE file for details.
