# Research Assistant AI

An advanced, multi-agent document analysis platform built with **LlamaIndex Workflows** and **LlamaCloud Managed RAG**. This system provides sophisticated research capabilities that go beyond simple retrieval, using intelligent planning, parallel execution, and comprehensive synthesis to answer complex questions about research papers and technical documentation.

## Key Features

- **Intelligent Research Planning**: Automatically decomposes complex questions into 2-4 focused research sub-tasks, executing them in parallel for faster results
- **Managed RAG Infrastructure**: Fully managed vector indexing via LlamaCloud - no local database setup required
- **Deep Document Extraction**: Integrated with LlamaParse for high-resolution agentic parsing of complex PDFs with tables, charts, and hierarchical structures
- **Precision Citations**: Automatic extraction of source documents with page numbers and relevance scores
- **Multi-Turn Chat**: Conversational mode with memory and context retention, plus file filtering for targeted queries
- **Real-Time Updates**: Live progress indicators during research execution via event streaming

## Technical Architecture

- **Orchestration**: LlamaIndex Workflows (Event-driven asynchronous execution)
- **Intelligence**: Google Gemini 2.5 Flash (Advanced reasoning & synthesis)
- **Data Infrastructure**: LlamaCloud Managed Index (Cloud-based vector store)
- **Parsing**: LlamaParse (Agentic mode with high-res OCR)
- **Frontend**: Streamlit (Reactive UI with custom CSS styling)

## Project Structure

```
research_paper_chatbot/
├── research_paper_chatbot/          # Main package
│   ├── __init__.py                  # Package initialization
│   ├── app.py                       # Streamlit frontend (entry point)
│   └── workflow.py                   # Workflow implementation
├── requirements.txt
├── README.md
├── run.sh                            # Launch script
└── setup_env.py                      # Configuration utility
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [Google Gemini API Key](https://aistudio.google.com/)
- [LlamaCloud API Key](https://cloud.llamaindex.ai/)

### Installation

1. Clone the repository and navigate to the `research_paper_chatbot` directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   LLAMA_API_KEY=your_llama_api_key_here
   LLAMA_CLOUD_PROJECT=Default
   ```

### Execution

Launch the application using Streamlit:
```bash
streamlit run research_paper_chatbot/app.py
```

Alternatively, use the provided helper script:
```bash
bash run.sh
```

## Usage

Upload PDF, DOCX, PPTX, XLSX, TXT, or MD files directly in the chat interface. Select specific documents from the sidebar to filter queries. Ask questions or use suggested prompts to begin your research analysis.

## Programmatic API

```python
from research_paper_chatbot.workflow import SyncDocumentChatbot

# Initialize
chatbot = SyncDocumentChatbot(
    google_api_key="your_key",
    llama_api_key="your_key"
)

# Upload documents
chatbot.upload_document("paper.pdf")

# Query documents
response = chatbot.query("What are the main findings?")

# Multi-turn chat with file filtering
result = chatbot.chat(
    message="Summarize the methodology",
    conversation_id="conv_123",
    file_ids=["file_id_1", "file_id_2"]
)
```
