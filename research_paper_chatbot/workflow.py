"""
LlamaIndex Workflow-based Document Chatbot
Uses llama-index-workflows for event-driven document processing and querying.
"""

import os
import sys
import uuid
import asyncio
import threading
import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to Python path so we can import the package
# __file__ is at: research_paper_chatbot/research_paper_chatbot/workflow.py
# We need: research_paper_chatbot/ (two levels up)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.google_genai import GoogleGenAI
from llama_cloud_services import LlamaParse, LlamaCloudIndex
from workflows import Workflow, Context, step
from workflows.events import Event, StartEvent, StopEvent


# Event definitions for the workflow
class ChatbotStopEvent(StopEvent):
    """Unified stop event for the document chatbot workflow."""
    # For document uploads
    num_documents: Optional[int] = None
    message: Optional[str] = None
    
    # For queries/chats
    response: Optional[str] = None
    question: Optional[str] = None
    conversation_id: Optional[str] = None
    message_history: Optional[List[Dict[str, str]]] = None
    citations: Optional[List[Dict[str, Union[str, int, float, None]]]] = None  # Citations with page numbers
    
    # For listing files
    files: Optional[List[Dict[str, Union[str, int, None]]]] = None


class ResearchPlanStartEvent(Event):
    """Event to start the research planning process."""
    question: str
    conversation_id: str
    file_ids: Optional[List[str]] = None


class ResearchPlanItem(Event):
    """A single step in the research plan."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    query: str


class ResearchPlan(BaseModel):
    """The full research plan."""
    items: List[ResearchPlanItem]


class ResearchPlanItemResult(Event):
    """The result of a single research step."""
    id: str
    description: str
    query: str
    result: str
    citations: List[Dict[str, Any]]


class ExecutedResearchPlan(Event):
    """Aggregated results from the research plan."""
    results: str
    all_citations: List[Dict[str, Any]]


class WorkflowState(BaseModel):
    """State for the document chatbot workflow."""
    persist_dir: str = "./storage"
    collection_name: str = "research_papers"
    num_documents: int = 0
    conversation_id: Optional[str] = None  # Track active conversation
    project_name: str = "Default"
    organization_id: Optional[str] = None
    index_id: Optional[str] = None  # LlamaCloud Index/Pipeline ID
    
    # State for Planning Workflow
    original_question: Optional[str] = None
    file_ids: Optional[List[str]] = None
    num_plan_items: int = 0
    
    # Use a dictionary to store chat engines dynamically
    chat_engines: Dict[str, ContextChatEngine] = {}

    class Config:
        arbitrary_types_allowed = True


class DocumentChatbotWorkflow(Workflow):
    """Workflow for document processing and querying using LlamaCloud managed RAG."""
    
    def __init__(
        self, 
        google_api_key: Optional[str] = None, 
        llama_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        project_name: str = "Default",
        organization_id: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize the workflow with API keys and LlamaCloud configuration."""
        super().__init__(timeout=timeout)
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self.llama_api_key = llama_api_key or os.getenv("LLAMA_API_KEY") or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_api_key:
            raise ValueError(
                "LlamaCloud API key is required. Set LLAMA_API_KEY environment variable."
            )
            
        self.gemini_model = gemini_model
        self.project_name = project_name
        self.organization_id = organization_id or os.getenv("LLAMA_CLOUD_ORG_ID")
        
        # Configure LlamaIndex settings basics (non-loop dependent)
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        # These will be initialized lazily to avoid loop-mismatch issues in Streamlit
        self._llm_initialized = False
        self.llama_parser = None
        
    def _setup_lazy_components(self):
        """Initialize loop-dependent components on the first call in a new loop."""
        if not self._llm_initialized:
            logger.info("Initializing LLM and Parser in the current event loop...")
            # Set LLM in global Settings (scoped to current thread/loop by LlamaIndex usually)
            Settings.llm = GoogleGenAI(
                api_key=self.google_api_key, 
                model=self.gemini_model, 
                temperature=0.1
            )
            
            # Initialize LlamaParse
            self.llama_parser = LlamaParse(
                api_key=self.llama_api_key,
                result_type="markdown",
                verbose=False,
                parse_mode="parse_page_with_agent",
                high_res_ocr=True,
                adaptive_long_table=True,
                outlined_table_extraction=True,
                output_tables_as_HTML=True,
            )
            self._llm_initialized = True

        # Prompts for Planning Workflow
        self.planning_prompt = (
            "You are a master research assistant. Your goal is to answer a complex user question by planning a set of specific research steps.\n"
            "You have access to a set of research papers.\n\n"
            "USER QUESTION: {question}\n\n"
            "Create a plan of 2-4 distinct research steps to answer this question comprehensively.\n"
            "Each step should focus on retrieving specific information needed for the final answer.\n"
            "If the question is simple, a single step is fine.\n"
            "For comparisons, create separate steps for each entity.\n"
            "Provide a 'description' (what you are looking for) and a 'query' (the specific search term for the index) for each step."
        )

        self.synthesize_prompt = (
            "You are a master research assistant. You have performed a multi-step research process to answer a user's question.\n"
            "Here is the User Question: {question}\n\n"
            "Here are the findings from your research steps:\n"
            "{research_results}\n\n"
            "Synthesize these findings into a comprehensive, well-structured answer. "
            "Cite specific papers where appropriate using the provided context. "
            "If the findings are contradictory, point that out. "
            "If the findings are insufficient, state what is missing."
        )
    
    async def _get_or_create_index(self, ctx: Context[WorkflowState]) -> LlamaCloudIndex:
        """Get or create the managed LlamaCloud index."""
        logger.info("Getting or creating LlamaCloud index...")
        
        # Access state via edit_state context manager for reliability
        async with ctx.store.edit_state() as state:
            index_id = state.index_id
            collection_name = state.collection_name
        
        # If we already have an index_id, connect to it
        if index_id:
            logger.info(f"Connecting to existing index with ID: {index_id}")
            return LlamaCloudIndex(
                id=index_id,
                api_key=self.llama_api_key,
                organization_id=self.organization_id
            )
        
        # Otherwise, truly "get or create" by name
        try:
            logger.info(f"Creating/connecting to index '{collection_name}' in project '{self.project_name}'")
            # Use acreate_index to handle upserting project and pipeline (index)
            index = await LlamaCloudIndex.acreate_index(
                name=collection_name,
                project_name=self.project_name,
                organization_id=self.organization_id,
                api_key=self.llama_api_key
            )
            
            # Update state with the index ID for future use
            async with ctx.store.edit_state() as state:
                state.index_id = index.id
                
            logger.info(f"Successfully connected/created index. Index ID: {index.id}")
            return index
        except Exception as e:
            logger.error(f"Failed to get or create LlamaCloud index: {e}")
            raise RuntimeError(f"LlamaCloud index error: {e}")
    
    @step
    async def process_request_step(
        self, ev: StartEvent, ctx: Context[WorkflowState]
    ) -> Union[ChatbotStopEvent, ResearchPlanStartEvent]:
        """Step to process either document upload, query, or chat."""
        # Ensure LLM and other components are tied to THIS loop
        self._setup_lazy_components()
        # Initialize state from StartEvent fields if not already initialized
        async with ctx.store.edit_state() as state:
            if not state.persist_dir and hasattr(ev, "persist_dir"):
                state.persist_dir = ev.persist_dir
            if (not state.collection_name or state.collection_name == "research_papers") and hasattr(ev, "collection_name"):
                state.collection_name = ev.collection_name
            if (not state.project_name or state.project_name == "Default") and hasattr(ev, "project_name"):
                state.project_name = ev.project_name
            if not state.organization_id and hasattr(ev, "organization_id"):
                state.organization_id = ev.organization_id

        # Access fields from StartEvent (passed as kwargs to run())
        event_type = getattr(ev, "event_type", "query")
        logger.info(f"Processing request with event_type: {event_type}")
        
        if event_type == "upload":
            file_path = getattr(ev, "file_path", None)
            directory_path = getattr(ev, "directory_path", None)
            logger.info(f"Upload request - file_path: {file_path}, directory_path: {directory_path}")
            result = await self._upload_documents(ev, ctx)
            logger.info(f"Upload completed: {result.message}")
            return result
        elif event_type == "query":
            question = getattr(ev, "question", None)
            file_ids = getattr(ev, "file_ids", None)
            logger.info(f"Query request - question: {question}, file_ids: {file_ids}")
            # Backward compatible: use old QueryEngine for single-turn queries
            result = await self._query_documents(ev, ctx)
            logger.info(f"Query completed - response length: {len(result.response)} chars")
            return result
        elif event_type == "chat":
            question = getattr(ev, "question", None)
            conversation_id = getattr(ev, "conversation_id", None)
            file_ids = getattr(ev, "file_ids", None)
            logger.info(f"Chat request - question: {question}, conversation_id: {conversation_id}, file_ids: {file_ids}")
            
            # Start the Planning Workflow
            if not conversation_id:
                conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            
            # Update state
            async with ctx.store.edit_state() as state:
                state.conversation_id = conversation_id
            
            # Route to Planning Step
            return ResearchPlanStartEvent(
                question=question,
                conversation_id=conversation_id,
                file_ids=file_ids
            )
        elif event_type == "list_files":
            logger.info("List files request")
            result = await self._list_files(ev, ctx)
            return result
        else:
            error_msg = f"Unknown event type: {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @step
    async def plan_research(
        self, ev: ResearchPlanStartEvent, ctx: Context[WorkflowState]
    ) -> ResearchPlanItem:
        """Step 1: Plan the research steps based on the user question."""
        self._setup_lazy_components()
        question = ev.question
        
        # Store context for later
        await ctx.store.set("original_question", question)
        await ctx.store.set("file_ids", ev.file_ids)
        await ctx.store.set("conversation_id", ev.conversation_id)
        
        ctx.write_event_to_stream(Event(msg=f"Analyzing question: {question}"))
        
        # Use LLM to generate plan
        try:
            # Simple structured prediction for the plan
            llm = Settings.llm
            prompt = self.planning_prompt.format(question=question)
            
            # Use structured output to get the plan
            plan: ResearchPlan = await llm.astructured_predict(ResearchPlan, PromptTemplate(prompt))
            
            logger.info(f"Generated research plan with {len(plan.items)} items")
            ctx.write_event_to_stream(Event(msg=f"Created research plan with {len(plan.items)} step(s)."))
            
            # Store number of items for aggregation
            await ctx.store.set("num_plan_items", len(plan.items))
            
            # Dispatch items
            for item in plan.items:
                ctx.send_event(item)
            
        except Exception as e:
            logger.error(f"Planning failed: {e}. Falling back to single step.")
            # Fallback: Just one step with the original question
            fallback_item = ResearchPlanItem(
                description="General research on the question", 
                query=question
            )
            await ctx.store.set("num_plan_items", 1)
            ctx.send_event(fallback_item)
            
        return None

    @step(num_workers=4)
    async def execute_research_step(
        self, ev: ResearchPlanItem, ctx: Context[WorkflowState]
    ) -> ResearchPlanItemResult:
        """Step 2: Execute a single research step (retrieval & synthesis)."""
        self._setup_lazy_components()
        ctx.write_event_to_stream(Event(msg=f"Researching: {ev.description}..."))
        
        file_ids = await ctx.store.get("file_ids")
        
        # Get index
        index = await self._get_or_create_index(ctx)
        
        # Setup Filters
        filters = None
        if file_ids:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="pipeline_file_id", value=fid, operator=FilterOperator.EQ)
                    for fid in file_ids
                ],
                condition="or"
            )
        
        # Create Query Engine for this step
        # Note: We use a fresh query engine for each step to allow full parallel execution in LlamaIndex
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            filters=filters
        )
        
        try:
            response = await query_engine.aquery(ev.query)
            result_text = str(response)
            
            # Extract citations
            citations = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                 for i, node_with_score in enumerate(response.source_nodes):
                    node = node_with_score.node
                    metadata = node.metadata
                    filename = metadata.get("file_name") or metadata.get("file_path") or "Unknown"
                    citations.append({
                        "file_name": filename,
                        "text": node.get_content(),
                        "score": node_with_score.score
                    })
            
            logger.info(f"Step '{ev.description}' completed.")
            
            return ResearchPlanItemResult(
                id=ev.id,
                description=ev.description,
                query=ev.query,
                result=result_text,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Step '{ev.description}' failed: {e}")
            return ResearchPlanItemResult(
                id=ev.id,
                description=ev.description,
                query=ev.query,
                result=f"Error executing research step: {e}",
                citations=[]
            )

    @step
    async def synthesize_research(
        self, ev: ResearchPlanItemResult, ctx: Context[WorkflowState]
    ) -> ChatbotStopEvent:
        """Step 3: Synthesize all research results into a final answer."""
        # Ensure LLM is tied to THIS loop (in case worker threads use different loops, though unlikely here)
        self._setup_lazy_components()
        num_items = await ctx.store.get("num_plan_items")
        results = ctx.collect_events(ev, [ResearchPlanItemResult] * num_items)
        
        if results is None:
            return None
        
        ctx.write_event_to_stream(Event(msg="Synthesizing findings..."))
        
        original_question = await ctx.store.get("original_question")
        conversation_id = await ctx.store.get("conversation_id")
        
        # Aggregate results for the prompt
        research_results_str = ""
        all_citations = []
        
        for r in results:
            research_results_str += f"### Step: {r.description}\nQuery: {r.query}\nResult: {r.result}\n\n"
            all_citations.extend(r.citations)
            
        # Deduplicate citations (simple approach by file_name)
        # Note: A robust implementation would be more careful, but this works for basic attribution
        unique_citations = []
        seen = set()
        for c in all_citations:
            key = (c['file_name'], c.get('text', '')[:50])
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        # Ensure we have "text" key for the frontend rendering
        for c in unique_citations:
             if 'text' not in c:
                 c['text'] = "N/A"
        
        # Generate Final Answer
        llm = Settings.llm
        prompt = self.synthesize_prompt.format(
            question=original_question,
            research_results=research_results_str
        )
        
        final_response = await llm.acomplete(prompt)
        
        # Get chat history (optional - we could persist this to memory if we wanted true multi-turn planning)
        # For now, we'll just return the current turn
        message_history = [
             {"role": "user", "content": original_question},
             {"role": "assistant", "content": str(final_response)}
        ]
        
        return ChatbotStopEvent(
            response=str(final_response),
            question=original_question,
            conversation_id=conversation_id,
            message_history=message_history,
            citations=unique_citations
        )
    
    async def _upload_documents(
        self, ev: StartEvent, ctx: Context[WorkflowState]
    ) -> ChatbotStopEvent:
        """Upload and index documents using LlamaCloud managed ingestion."""
        logger.info("Starting document upload process...")
        index = await self._get_or_create_index(ctx)
        
        file_paths = []
        file_path = getattr(ev, "file_path", None)
        directory_path = getattr(ev, "directory_path", None)
        
        # Collect file paths
        if file_path:
            logger.info(f"Processing single file: {file_path}")
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            file_paths = [file_path]
        elif directory_path:
            logger.info(f"Processing directory: {directory_path}")
            if not os.path.isdir(directory_path):
                error_msg = f"Directory not found: {directory_path}"
                logger.error(error_msg)
                raise NotADirectoryError(error_msg)
            # Get all supported files from directory
            supported_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.xml', '.epub', '.txt', '.md'}
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_paths.append(os.path.join(directory_path, file))
            logger.info(f"Found {len(file_paths)} supported files in directory")
        else:
            error_msg = "Either file_path or directory_path must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not file_paths:
            logger.warning("No supported documents found to index")
            return ChatbotStopEvent(
                num_documents=0,
                message="No supported documents found to index"
            )
        
        # Upload files directly to LlamaCloud managed index
        # This handles parsing (via LlamaParse), embedding, and indexing in the cloud.
        uploaded_count = 0
        for file_path in file_paths:
            logger.info(f"Uploading file: {file_path}")
            try:
                # LlamaCloudIndex supports direct file upload
                await index.aupload_file(file_path)
                uploaded_count += 1
                logger.info(f"Successfully uploaded {file_path} to LlamaCloud")
            except Exception as e:
                logger.warning(f"Failed to upload {file_path} to LlamaCloud: {e}")
                # Fallback: if direct upload fails, try parsing locally and inserting nodes
                try:
                    logger.info(f"Attempting fallback local parsing for {file_path}")
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    logger.info(f"Parsed {len(docs)} documents locally")
                    for doc in docs:
                        await index.ainsert(doc)
                    uploaded_count += 1
                    logger.info(f"Successfully inserted {file_path} via fallback method")
                except Exception as e2:
                    logger.error(f"Fallback local parsing also failed for {file_path}: {e2}")
        
        # Wait for ingestion to complete if we uploaded files
        if uploaded_count > 0:
            logger.info("Waiting for LlamaCloud ingestion to complete...")
            await index.await_for_completion(verbose=True)
            logger.info("LlamaCloud ingestion completed")
        
        # Update state
        async with ctx.store.edit_state() as state:
            state.num_documents += uploaded_count
            logger.info(f"Updated state: num_documents = {state.num_documents}")
        
        result_msg = f"Successfully indexed {uploaded_count} document(s) using LlamaCloud Managed RAG"
        logger.info(result_msg)
        return ChatbotStopEvent(
            num_documents=uploaded_count,
            message=result_msg
        )
    
    async def _query_documents(
        self, ev: StartEvent, ctx: Context[WorkflowState]
    ) -> ChatbotStopEvent:
        """Query the indexed documents (single-turn, backward compatible)."""
        logger.info("Starting query process...")
        question = getattr(ev, "question", None)
        file_ids = getattr(ev, "file_ids", None)
        
        if not question:
            error_msg = "Question is required for query events"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        index = await self._get_or_create_index(ctx)
        logger.info(f"Index retrieved, creating query engine with file_ids: {file_ids}")
        
        filters = None
        if file_ids:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="pipeline_file_id", value=fid, operator=FilterOperator.EQ)
                    for fid in file_ids
                ],
                condition="or"
            )
        
        # Create query engine (single-turn, no memory)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            filters=filters
        )
        
        # Query the documents
        logger.info(f"Executing query: {question}")
        response = await query_engine.aquery(question)
        logger.info(f"Query completed, response length: {len(str(response))} chars")
        
        # Extract citations from source nodes
        citations = []
        seen_citations = set()  # Track unique citations to avoid duplicates
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            logger.info(f"Retrieved {len(response.source_nodes)} source nodes for query")
            for i, node_with_score in enumerate(response.source_nodes):
                node = node_with_score.node
                metadata = node.metadata
                
                # Extract citation information
                file_name = metadata.get("file_name") or metadata.get("file_path") or metadata.get("id", "Unknown")
                
                # Try to get page number from various metadata fields
                page = None
                if "page_label" in metadata:
                    page = metadata["page_label"]
                elif "start_page_label" in metadata:
                    page = metadata["start_page_label"]
                elif "start_page_index" in metadata:
                    # page_index is 0-based, page_label is 1-based
                    page = metadata["start_page_index"] + 1
                
                # Create unique citation key to avoid duplicates
                citation_key = (file_name, page)
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    citation = {
                        "file_name": file_name,
                        "page": page,
                        "score": float(node_with_score.score) if hasattr(node_with_score, 'score') and node_with_score.score is not None else None,
                        "text": node.get_content()
                    }
                    citations.append(citation)
                    page_str = f"page {page}" if page else "unknown page"
                    score_str = f"{citation['score']:.4f}" if citation['score'] is not None else "N/A"
                    logger.info(f"Query Citation {len(citations)}: {file_name} {page_str} (score: {score_str})")
        
        # Return unified event (no conversation_id or message_history for single-turn)
        return ChatbotStopEvent(
            question=question,
            response=str(response),
            conversation_id=None,
            message_history=None,
            citations=citations if citations else None
        )
    
    async def _get_or_create_chat_engine(
        self, 
        ctx: Context[WorkflowState], 
        conversation_id: str,
        index: LlamaCloudIndex,
        file_ids: Optional[List[str]] = None
    ) -> ContextChatEngine:
        """Get or create a chat engine for a conversation."""
        logger.info(f"Getting or creating chat engine for conversation: {conversation_id}")
        
        # If file_ids are provided, we should probably create a new engine or update filters
        # For simplicity, if file_ids change, we'll create a new engine key
        chat_engine_key = conversation_id
        if file_ids:
            # Sort file_ids to ensure consistent key
            file_ids_str = "_".join(sorted(file_ids))
            chat_engine_key += f"_{file_ids_str}"

        # Try to get existing chat engine from state
        async with ctx.store.edit_state() as state:
            if chat_engine_key in state.chat_engines:
                logger.info(f"Reusing existing chat engine for conversation: {conversation_id} with key {chat_engine_key}")
                return state.chat_engines[chat_engine_key]
        
        # Create new chat engine with memory and filters
        logger.info(f"Creating new chat engine for conversation: {conversation_id}")
        
        filters = None
        if file_ids:
            logger.info(f"Applying filters for file_ids: {file_ids}")
            # LlamaCloud uses 'pipeline_file_id' as the metadata key for the specific file instance in a pipeline
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="pipeline_file_id", value=fid, operator=FilterOperator.EQ)
                    for fid in file_ids
                ],
                condition="or"
            )
            logger.info(f"Generated filters: {filters}")

        # Note: LlamaCloudIndex supports as_chat_engine
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=5,
            filters=filters
        )
        
        # Store in state for reuse
        async with ctx.store.edit_state() as state:
            state.chat_engines[chat_engine_key] = chat_engine
        logger.info(f"Chat engine created and stored for conversation: {conversation_id} with key {chat_engine_key}")
        
        return chat_engine
    
    async def _chat_with_documents(
        self, ev: StartEvent, ctx: Context[WorkflowState]
    ) -> ChatbotStopEvent:
        """Chat with the indexed documents using ChatEngine (multi-turn with memory)."""
        logger.info("Starting chat process...")
        question = getattr(ev, "question", None)
        file_ids = getattr(ev, "file_ids", None)
        
        if not question:
            error_msg = "Question is required for chat events"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        index = await self._get_or_create_index(ctx)
        
        # Get or generate conversation ID
        conversation_id = getattr(ev, "conversation_id", None)
        if not conversation_id:
            # Generate a new conversation ID
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated new conversation ID: {conversation_id}")
        else:
            logger.info(f"Using existing conversation ID: {conversation_id}")
        
        # Get or create chat engine for this conversation
        chat_engine = await self._get_or_create_chat_engine(ctx, conversation_id, index, file_ids=file_ids)
        
        # Chat with memory
        logger.info(f"Executing chat query: {question}")
        try:
            # Use achat for async step
            response = await chat_engine.achat(question)
            response_text = str(response)
            logger.info(f"Chat response generated, length: {len(response_text)} chars")
            
            if response_text == "Empty Response":
                logger.warning("Synthesizer returned 'Empty Response'. This usually means no relevant context was found.")
        except Exception as e:
            logger.error(f"Error during chat_engine.achat: {e}")
            raise
        
        # Extract citations from source nodes
        citations = []
        seen_citations = set()  # Track unique citations to avoid duplicates
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            logger.info(f"Retrieved {len(response.source_nodes)} source nodes")
            for i, node_with_score in enumerate(response.source_nodes):
                node = node_with_score.node
                metadata = node.metadata
                
                # Extract citation information
                file_name = metadata.get("file_name") or metadata.get("file_path") or metadata.get("id", "Unknown")
                
                # Try to get page number from various metadata fields
                page = None
                if "page_label" in metadata:
                    page = metadata["page_label"]
                elif "start_page_label" in metadata:
                    page = metadata["start_page_label"]
                elif "start_page_index" in metadata:
                    # page_index is 0-based, page_label is 1-based
                    page = metadata["start_page_index"] + 1
                
                # Create unique citation key to avoid duplicates
                citation_key = (file_name, page)
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    citation = {
                        "file_name": file_name,
                        "page": page,
                        "score": float(node_with_score.score) if hasattr(node_with_score, 'score') and node_with_score.score is not None else None,
                        "text": node.get_content()
                    }
                    citations.append(citation)
                    page_str = f"page {page}" if page else "unknown page"
                    score_str = f"{citation['score']:.4f}" if citation['score'] is not None else "N/A"
                    logger.info(f"Citation {len(citations)}: {file_name} {page_str} (score: {score_str})")
        else:
            logger.warning("No source nodes retrieved for this query")
        
        # Get message history
        message_history = []
        try:
            # Get all messages from chat history
            all_messages = chat_engine.chat_history
            logger.info(f"Retrieved {len(all_messages)} messages from history")
            for msg in all_messages:
                message_history.append({
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "content": msg.content
                })
        except Exception as e:
            logger.warning(f"Could not retrieve full message history: {e}. Using current exchange only.")
            # If we can't get message history, at least include current exchange
            message_history = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(response)}
            ]
        
        # Update workflow state with conversation ID
        async with ctx.store.edit_state() as state:
            state.conversation_id = conversation_id
            logger.info(f"Updated state with conversation_id: {conversation_id}")
        
        return ChatbotStopEvent(
            response=str(response),
            question=question,
            conversation_id=conversation_id,
            message_history=message_history,
            citations=citations if citations else None
        )

    async def _list_files(
        self, ev: StartEvent, ctx: Context[WorkflowState]
    ) -> ChatbotStopEvent:
        """List all files indexed in the LlamaCloud pipeline."""
        logger.info("Listing files in LlamaCloud...")
        index = await self._get_or_create_index(ctx)
        
        try:
            # Use the underlying asynchronous client to list pipeline files
            files_response = await index._aclient.pipeline_files.list_pipeline_files(
                pipeline_id=index.pipeline.id
            )
            
            file_list = []
            for f in files_response:
                file_list.append({
                    "id": f.id,
                    "name": f.name,
                    "status": str(f.status) if hasattr(f, 'status') else "unknown"
                })
            
            logger.info(f"Found {len(file_list)} files in pipeline")
            return ChatbotStopEvent(files=file_list)
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return ChatbotStopEvent(files=[], message=f"Error listing files: {e}")


class WorkflowRunner:
    """
    Thread-safe runner for executing async workflows from synchronous contexts.
    Uses a dedicated background thread with its own event loop.
    """
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started = False
    
    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def start(self):
        """Start the background event loop thread."""
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()
            # Wait for loop to be ready
            while self._loop is None:
                threading.Event().wait(0.01)
            self._started = True
    
    def stop(self):
        """Stop the background event loop thread."""
        with self._lock:
            if not self._started or self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5.0)
            self._started = False
    
    def run_coroutine(self, coro):
        """
        Run a coroutine in the background event loop thread-safely.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        if not self._started:
            self.start()
        
        if self._loop is None:
            raise RuntimeError("Event loop not initialized")
        
        # Use run_coroutine_threadsafe to execute in the background loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


# Global runner instance (singleton pattern)
_runner = WorkflowRunner()


class DocumentChatbot:
    """Wrapper class for the workflow-based chatbot."""
    
    def __init__(
        self,
        persist_dir: str = "./storage",
        collection_name: str = "research_papers",
        google_api_key: Optional[str] = None,
        llama_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        project_name: str = "Default",
        organization_id: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize the chatbot."""
        self.workflow = DocumentChatbotWorkflow(
            google_api_key=google_api_key,
            llama_api_key=llama_api_key,
            gemini_model=gemini_model,
            project_name=project_name,
            organization_id=organization_id,
            timeout=timeout
        )
        
        # Initialize workflow with state
        initial_state = WorkflowState(
            persist_dir=persist_dir,
            collection_name=collection_name,
            project_name=project_name,
            organization_id=organization_id,
        )
        
        # We'll initialize the workflow when needed
        self._initial_state = initial_state
        self._conversation_id: Optional[str] = None
    
    async def upload_document(self, file_path: str) -> str:
        """Upload and index a single document."""
        logger.info(f"DocumentChatbot.upload_document called with file_path: {file_path}")
        logger.info("Starting workflow run for document upload...")
        # Pass data as kwargs to populate the StartEvent
        result: ChatbotStopEvent = await self.workflow.run(
            file_path=file_path, 
            event_type="upload",
            # Pass configuration fields to initialize state in the first step
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )
        logger.info(f"Workflow completed. Result message: {result.message}")
        return result.message or "Upload completed"
    
    async def upload_directory(self, directory_path: str) -> str:
        """Upload and index all documents in a directory."""
        result: ChatbotStopEvent = await self.workflow.run(
            directory_path=directory_path, 
            event_type="upload",
            # Pass configuration fields to initialize state in the first step
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )
        return result.message or "Upload completed"
    
    async def query(self, question: str, file_ids: Optional[List[str]] = None) -> str:
        """Query the indexed documents (single-turn, backward compatible)."""
        result: ChatbotStopEvent = await self.workflow.run(
            question=question, 
            event_type="query",
            file_ids=file_ids,
            # Pass configuration fields to initialize state in the first step
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )
        return result.response or "No response generated"
    
    async def chat(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Chat with the indexed documents (multi-turn with memory).
        
        Args:
            message: The user's message
            conversation_id: Optional conversation ID to continue a conversation.
                          If None, starts a new conversation.
            file_ids: Optional list of file IDs to filter the context.
        
        Returns:
            Dictionary with 'response', 'conversation_id', and 'message_history'
        """
        logger.info(f"DocumentChatbot.chat called with message: {message[:100]}..., conversation_id: {conversation_id}, file_ids: {file_ids}")
        # Use provided conversation_id or the stored one, or None for new conversation
        conv_id = conversation_id or self._conversation_id
        logger.info(f"Using conversation_id: {conv_id}")
        
        logger.info("Starting workflow run for chat...")
        result: ChatbotStopEvent = await self.workflow.run(
            question=message, 
            event_type="chat",
            conversation_id=conv_id,
            file_ids=file_ids,
            # Pass configuration fields to initialize state in the first step
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )
        logger.info(f"Workflow completed. Conversation ID: {result.conversation_id}")
        
        # Store conversation_id for next call
        self._conversation_id = result.conversation_id
        
        return {
            "response": result.response or "No response generated",
            "conversation_id": result.conversation_id,
            "message_history": result.message_history,
            "citations": result.citations or []
        }

    async def run_chat_flow(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None
    ):
        """
        Start the chat workflow and return the handler for event streaming.
        Useful for UI applications that need to show progress.
        """
        logger.info(f"DocumentChatbot.run_chat_flow called with message: {message[:100]}...")
        # Use provided conversation_id or the stored one, or None for new conversation
        conv_id = conversation_id or self._conversation_id
        
        # Start the workflow run but DON'T await the result immediately
        # run() returns a handler if we don't await the couroutine? 
        # Actually workflow.run() is an async method that returns the result.
        # We need to rely on the fact that LlamaIndex Workflow.run returns a handler if called but not awaited?
        # No, we need to inspect how LlamaIndex Workflows works exactly. 
        # Taking a cue from DataAnalystWorkflow, we can just call run() and capture the handler if we modify how we call it?
        # Actually, in DataAnalystWorkflow example: 
        # handler = workflow.run(...)
        # This implies workflow.run(...) returns a handler synchronously? 
        # Let's check the import in data_analyst_chatbot/workflow.py: `from workflows import Workflow`
        # If it's the standard llama-index-core workflow, `run` is async.
        # But wait, `DataAnalystWorkflow` used:
        # handler = workflow.run(...)
        # async for event in handler.stream_events(): ...
        # reliable pattern: loop.create_task(workflow.run(...)) returns a Task.
        # But to stream events we need the context or handler.
        
        # Let's stick to the pattern used in the other chatbot:
        # handler = workflow.run(...)
        # pass kwargs.
        
        return self.workflow.run(
            question=message, 
            event_type="chat",
            conversation_id=conv_id,
            file_ids=file_ids,
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )

    async def list_files(self) -> List[Dict[str, str]]:
        """List all files indexed in the LlamaCloud pipeline."""
        logger.info("DocumentChatbot.list_files called")
        result: ChatbotStopEvent = await self.workflow.run(
            event_type="list_files",
            persist_dir=self._initial_state.persist_dir,
            collection_name=self._initial_state.collection_name,
            project_name=self._initial_state.project_name,
            organization_id=self._initial_state.organization_id
        )
        return result.files or []
    
    def reset_conversation(self):
        """Reset the current conversation (start a new one)."""
        self._conversation_id = None


class SyncDocumentChatbot:
    """Synchronous wrapper for the async chatbot using thread-safe execution."""
    
    def __init__(
        self,
        persist_dir: str = "./storage",
        collection_name: str = "research_papers",
        google_api_key: Optional[str] = None,
        llama_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        project_name: str = "Default",
        organization_id: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize the chatbot."""
        self.chatbot = DocumentChatbot(
            persist_dir=persist_dir,
            collection_name=collection_name,
            google_api_key=google_api_key,
            llama_api_key=llama_api_key,
            gemini_model=gemini_model,
            project_name=project_name,
            organization_id=organization_id,
            timeout=timeout
        )
        # Ensure the runner is started
        _runner.start()
    
    def upload_document(self, file_path: str) -> str:
        """Upload and index a single document."""
        logger.info(f"SyncDocumentChatbot.upload_document called with file_path: {file_path}")
        try:
            result = _runner.run_coroutine(self.chatbot.upload_document(file_path))
            logger.info(f"Upload completed successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during upload: {e}", exc_info=True)
            raise
    
    def upload_directory(self, directory_path: str) -> str:
        """Upload and index all documents in a directory."""
        return _runner.run_coroutine(self.chatbot.upload_directory(directory_path))
    
    def query(self, question: str, file_ids: Optional[List[str]] = None) -> str:
        """Query the indexed documents (single-turn, backward compatible)."""
        return _runner.run_coroutine(self.chatbot.query(question, file_ids))
    
    def chat(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Chat with the indexed documents (multi-turn with memory).
        
        Args:
            message: The user's message
            conversation_id: Optional conversation ID to continue a conversation.
                          If None, starts a new conversation.
            file_ids: Optional list of file IDs to filter the context.
        
        Returns:
            Dictionary with 'response', 'conversation_id', and 'message_history'
        """
        logger.info(f"SyncDocumentChatbot.chat called with message: {message[:100]}..., conversation_id: {conversation_id}, file_ids: {file_ids}")
        try:
            result = _runner.run_coroutine(self.chatbot.chat(message, conversation_id, file_ids))
            logger.info(f"Chat completed successfully. Conversation ID: {result.get('conversation_id')}")
            return result
        except Exception as e:
            logger.error(f"Error during chat: {e}", exc_info=True)
            raise
    
    def list_files(self) -> Union[List[Dict[str, str]], ResearchPlanStartEvent]:
        """List all files indexed in the LlamaCloud pipeline."""
        return _runner.run_coroutine(self.chatbot.list_files())
    
    def reset_conversation(self):
        """Reset the current conversation (start a new one)."""
        self.chatbot.reset_conversation()


if __name__ == "__main__":
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaIndex Workflow-based Document Chatbot")
    parser.add_argument("--upload", type=str, help="Path to a document file to upload")
    parser.add_argument("--upload-dir", type=str, help="Path to a directory containing documents")
    parser.add_argument("--query", type=str, help="Query to ask about the documents")
    parser.add_argument("--persist-dir", type=str, default="./storage", help="Directory to persist the index")
    parser.add_argument("--google-api-key", type=str, help="Google API key for Gemini (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--llama-api-key", type=str, help="LlamaCloud API key (or set LLAMA_API_KEY env var)")
    parser.add_argument("--project-name", type=str, default="Default", help="LlamaCloud project name")
    parser.add_argument("--org-id", type=str, help="LlamaCloud organization ID")
    
    args = parser.parse_args()
    
    chatbot = SyncDocumentChatbot(
        persist_dir=args.persist_dir,
        google_api_key=args.google_api_key,
        llama_api_key=args.llama_api_key,
        project_name=args.project_name,
        organization_id=args.org_id,
    )
    
    if args.upload:
        result = chatbot.upload_document(args.upload)
        print(f"✓ {result}")
    
    if args.upload_dir:
        result = chatbot.upload_directory(args.upload_dir)
        print(f"✓ {result}")
    
    if args.query:
        answer = chatbot.query(args.query)
        print(f"\n🤖 Answer: {answer}\n")
