# Path: mycdagent/src/mycdagent/tools/rag_tool.py

from crewai.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGQueryInput(BaseModel):
    """Input schema for RAG Query Tool."""
    query: str = Field(..., description="The question to search for in the knowledge base.")

class RAGTool(BaseTool):
    name: str = "Knowledge Base Query Tool"
    description: str = (
        "Use this tool to query the knowledge base for information related to your task. "
        "This tool searches through the available text documents and returns the most relevant information."
    )
    args_schema: Type[BaseModel] = RAGQueryInput
    retriever = None
    
    def __init__(self, knowledge_dir: str = "knowledge", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_dir = knowledge_dir
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the vector store and retriever."""
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
                        "metadata": {"source": os.path.basename(file_path)}
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
        try:
            vectordb = Qdrant.from_documents(
                documents=splits,
                embedding=embeddings,
                location=":memory:",
                collection_name="knowledge_store",
                force_recreate=True
            )
            
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
            return "Knowledge base is not initialized. Please check if text files exist in the knowledge directory."
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant information found in the knowledge base."
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                results.append(f"Source: {source}\n\nContent:\n{doc.page_content}\n")
            
            return "\n---\n".join(results)
        
        except Exception as e:
            return f"Error querying the knowledge base: {str(e)}"