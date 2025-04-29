from crewai.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class RAGQueryInput(BaseModel):
    """Input schema for RAG Query Tool."""
    query: str = Field(..., description="The question to search for in the knowledge base.")

class RAGTool(BaseTool):
    name: str = "Knowledge Base Query Tool"
    description: str = (
        "Use this tool to query the banking knowledge base for information related to your task. "
        "This tool searches through the available banking text documents and returns the most relevant information."
    )
    args_schema: Type[BaseModel] = RAGQueryInput
    retriever = None
    
    def __init__(self, knowledge_dir: str = "knowledge", db_path: str = "vector_db", 
                 collection_name: str = "banking_knowledge", force_recreate: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_dir = knowledge_dir
        self.db_path = db_path
        self.collection_name = collection_name
        self.force_recreate = force_recreate
        self._initialize_retriever()
    
    def _should_recreate_db(self):
        """Check if we should recreate the database"""
        # Always recreate if forced
        if self.force_recreate:
            return True
            
        # If the db directory doesn't exist, create it
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            return True
            
        # If we have a client, check if the collection exists
        try:
            client = QdrantClient(path=self.db_path)
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # If the collection doesn't exist, create it
            if self.collection_name not in collection_names:
                return True
                
            # Check if there are any knowledge files newer than the DB
            if not os.path.exists(os.path.join(self.db_path, "last_update.txt")):
                return True
                
            # Check if any knowledge files were modified after DB creation
            with open(os.path.join(self.db_path, "last_update.txt"), "r") as f:
                last_update = float(f.read().strip())
                
            text_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
            for file_path in text_files:
                if os.path.getmtime(file_path) > last_update:
                    return True
                    
            return False
        except Exception as e:
            print(f"Error checking database: {str(e)}")
            return True
    
    def _initialize_retriever(self):
        """Initialize the vector store and retriever."""
        try:
            # Check if we need to recreate the database
            if not self._should_recreate_db():
                print("Using existing vector database...")
                # Connect to existing database
                client = QdrantClient(path=self.db_path)
                
                # Initialize embeddings model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                
                # Load existing vector store
                vectordb = Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=embeddings
                )
                
                # Initialize retriever
                self.retriever = vectordb.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.5,
                        "score_threshold": 0.3
                    }
                )
                print(f"Successfully connected to existing vector database at {self.db_path}")
                return
            
            # Recreate the database
            print("Creating new vector database...")
            
            # Collect all text files from the knowledge directory
            text_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
            
            if not text_files:
                print(f"No text files found in {self.knowledge_dir}. Retriever not initialized.")
                return
            
            # Load and process documents
            documents = []
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        documents.append({
                            "page_content": content,
                            "metadata": {
                                "source": os.path.basename(file_path),
                                "file_path": file_path,
                                "modified_time": os.path.getmtime(file_path)
                            }
                        })
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
            
            if not documents:
                print("No documents successfully loaded. Retriever not initialized.")
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            splits = []
            for doc in documents:
                chunks = text_splitter.create_documents(
                    texts=[doc["page_content"]],
                    metadatas=[doc["metadata"]]
                )
                splits.extend(chunks)
            
            # Initialize embeddings model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # Create vector store
            client = QdrantClient(path=self.db_path)
            
            # Create or recreate the collection
            try:
                client.get_collection(self.collection_name)
                client.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, which is fine
                pass
                
            # Create the collection with the right vector size
            vector_size = len(embeddings.embed_query("test"))
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            # Create vector store
            vectordb = Qdrant.from_documents(
                documents=splits,
                embedding=embeddings,
                location=self.db_path,  # Disk-based storage
                collection_name=self.collection_name,
                force_recreate=True  # Create new collection
            )
            
            # Save the last update time
            with open(os.path.join(self.db_path, "last_update.txt"), "w") as f:
                import time
                f.write(str(time.time()))
            
            # Initialize retriever
            self.retriever = vectordb.as_retriever(
                search_type="mmr",  # Maximal Marginal Relevance
                search_kwargs={
                    "k": 5,  # Number of documents to return
                    "fetch_k": 10,  # More docs to consider initially
                    "lambda_mult": 0.5,  # Balance relevance and diversity
                    "score_threshold": 0.3  # Minimum similarity threshold
                }
            )
            print(f"Retriever successfully initialized with {len(splits)} chunks from {len(text_files)} files.")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
    
    def _run(self, query: str) -> str:
        """Run the RAG tool with the given query."""
        if not self.retriever:
            return "Banking knowledge base is not initialized. Please check if text files exist in the knowledge directory."
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant banking information found in the knowledge base."
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                results.append(f"Source: {source}\n\nContent:\n{doc.page_content}\n")
            
            return "\n---\n".join(results)
        
        except Exception as e:
            return f"Error querying the banking knowledge base: {str(e)}"