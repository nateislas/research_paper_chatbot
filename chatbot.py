"""
LlamaIndex Document Chatbot
A simple chatbot that can upload and query research papers using LlamaIndex.
"""

import os
from pathlib import Path
from typing import Optional, List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class DocumentChatbot:
    """A chatbot that can load documents and answer questions about them."""
    
    def __init__(
        self,
        persist_dir: str = "./storage",
        collection_name: str = "research_papers",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the chatbot.
        
        Args:
            persist_dir: Directory to persist the index
            collection_name: Name of the ChromaDB collection
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        
        # Set up OpenAI API key
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma_db"))
        
        # Try to load existing index
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            # Try to load existing index
            storage_path = self.persist_dir / "index_storage"
            if storage_path.exists():
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
                self.index = load_index_from_storage(storage_context)
                print(f"✓ Loaded existing index from {storage_path}")
            else:
                # Create a new empty index
                chroma_collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name
                )
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                print("✓ Created new index")
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")
            # Create a new index
            chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex([], storage_context=storage_context)
            print("✓ Created new index")
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )
    
    def upload_document(self, file_path: str) -> None:
        """
        Upload and index a single document.
        
        Args:
            file_path: Path to the document file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📄 Loading document: {file_path}")
        
        # Load document
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        # Add documents to index
        for doc in documents:
            self.index.insert(doc)
        
        # Persist index
        storage_path = self.persist_dir / "index_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(storage_path))
        
        print(f"✓ Document indexed successfully!")
    
    def upload_directory(self, directory_path: str) -> None:
        """
        Upload and index all documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        print(f"📁 Loading documents from: {directory_path}")
        
        # Load all documents
        reader = SimpleDirectoryReader(input_dir=directory_path)
        documents = reader.load_data()
        
        if not documents:
            print("⚠ No documents found in directory")
            return
        
        print(f"Found {len(documents)} document(s)")
        
        # Add documents to index
        for doc in documents:
            self.index.insert(doc)
        
        # Persist index
        storage_path = self.persist_dir / "index_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(storage_path))
        
        print(f"✓ All documents indexed successfully!")
    
    def query(self, question: str) -> str:
        """
        Query the indexed documents.
        
        Args:
            question: The question to ask about the documents
            
        Returns:
            The answer from the chatbot
        """
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Please upload documents first.")
        
        print(f"🤔 Question: {question}")
        print("💭 Thinking...")
        
        response = self.query_engine.query(question)
        
        return str(response)
    
    def chat(self):
        """Interactive chat mode."""
        print("\n" + "="*60)
        print("📚 LlamaIndex Document Chatbot")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'upload <file_path>' to upload a document")
        print("Type 'upload_dir <directory_path>' to upload all documents in a directory")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.startswith('upload '):
                    file_path = user_input[7:].strip()
                    try:
                        self.upload_document(file_path)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue
                
                if user_input.startswith('upload_dir '):
                    dir_path = user_input[11:].strip()
                    try:
                        self.upload_directory(dir_path)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue
                
                # Regular query
                answer = self.query(user_input)
                print(f"\n🤖 Bot: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaIndex Document Chatbot")
    parser.add_argument(
        "--upload",
        type=str,
        help="Path to a document file to upload",
    )
    parser.add_argument(
        "--upload-dir",
        type=str,
        help="Path to a directory containing documents to upload",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask about the documents",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./storage",
        help="Directory to persist the index (default: ./storage)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = DocumentChatbot(
            persist_dir=args.persist_dir,
            openai_api_key=args.api_key,
        )
        
        # Upload documents if specified
        if args.upload:
            chatbot.upload_document(args.upload)
        
        if args.upload_dir:
            chatbot.upload_directory(args.upload_dir)
        
        # Query if specified
        if args.query:
            answer = chatbot.query(args.query)
            print(f"\n🤖 Answer: {answer}\n")
        else:
            # Interactive mode
            chatbot.chat()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

