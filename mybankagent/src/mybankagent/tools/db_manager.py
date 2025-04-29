"""
Database Manager for Banking Knowledge Vector Database
This module provides utilities for managing the Qdrant vector database
used by the Banking Agent RAG system.
"""

import os
import glob
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionDescription
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BankingDBManager:
    """Manager for the Banking Knowledge Vector Database"""
    
    def __init__(self, 
                knowledge_dir: str = "knowledge", 
                db_path: str = "vector_db", 
                collection_name: str = "banking_knowledge",
                embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the Database Manager
        
        Args:
            knowledge_dir: Directory containing knowledge text files
            db_path: Directory for the vector database
            collection_name: Name of the collection in Qdrant
            embedding_model: HuggingFace model to use for embeddings
        """
        self.knowledge_dir = knowledge_dir
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create directories if they don't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
    def get_client(self) -> QdrantClient:
        """Get a Qdrant client for the database"""
        return QdrantClient(path=self.db_path)
        
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embeddings model"""
        return HuggingFaceEmbeddings(model_name=self.embedding_model)
        
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        try:
            client = self.get_client()
            collections = client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {str(e)}")
            return []
            
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a collection"""
        name = collection_name or self.collection_name
        try:
            client = self.get_client()
            info = client.get_collection(name)
            return {
                "name": name,
                "vectors_count": info.vectors_count,
                "status": "active",
                "vector_size": info.config.params.size
            }
        except Exception as e:
            return {
                "name": name,
                "status": "not found",
                "error": str(e)
            }
            
    def create_backup(self) -> str:
        """Create a backup of the current database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database directory not found: {self.db_path}")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.db_path}_backup_{timestamp}"
        
        try:
            shutil.copytree(self.db_path, backup_dir)
            return f"Backup created at {backup_dir}"
        except Exception as e:
            return f"Error creating backup: {str(e)}"
            
    def restore_backup(self, backup_path: str) -> str:
        """Restore database from a backup"""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup directory not found: {backup_path}")
            
        try:
            # Remove current DB if it exists
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                
            # Copy backup to current DB path
            shutil.copytree(backup_path, self.db_path)
            return f"Database restored from {backup_path}"
        except Exception as e:
            return f"Error restoring backup: {str(e)}"
            
    def delete_database(self) -> str:
        """Delete the entire vector database"""
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                return f"Database at {self.db_path} has been deleted"
            else:
                return f"Database at {self.db_path} does not exist"
        except Exception as e:
            return f"Error deleting database: {str(e)}"
            
    def rebuild_database(self, force: bool = True) -> Dict[str, Any]:
        """
        Rebuild the vector database from knowledge files
        
        Args:
            force: Whether to force rebuild even if no changes detected
            
        Returns:
            Dictionary with rebuild status and information
        """
        start_time = time.time()
        
        try:
            # Get text files
            text_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
            
            if not text_files:
                return {
                    "status": "error",
                    "message": f"No text files found in {self.knowledge_dir}",
                    "time_taken": time.time() - start_time
                }
                
            # Load documents
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
                return {
                    "status": "error",
                    "message": "No documents successfully loaded",
                    "time_taken": time.time() - start_time
                }
                
            # Split documents
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
                
            # Get embeddings
            embeddings = self.get_embeddings()
            
            # Get client
            client = self.get_client()
            
            # Check if collection exists and delete if needed
            try:
                collections = self.list_collections()
                if self.collection_name in collections:
                    client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                print(f"Error checking/deleting collection: {str(e)}")
                
            # Create collection
            vector_size = len(embeddings.embed_query("test"))
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            # Create vector store
            vectordb = Qdrant.from_documents(
                documents=splits,
                embedding=embeddings,
                location=self.db_path,
                collection_name=self.collection_name,
                force_recreate=True
            )
            
            # Save the last update time
            with open(os.path.join(self.db_path, "last_update.txt"), "w") as f:
                f.write(str(time.time()))
                
            time_taken = time.time() - start_time
                
            return {
                "status": "success",
                "message": f"Database rebuilt successfully with {len(splits)} chunks from {len(text_files)} files",
                "chunks": len(splits),
                "files": len(text_files),
                "time_taken": time_taken
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error rebuilding database: {str(e)}",
                "time_taken": time.time() - start_time
            }
            
    def get_retriever(self):
        """Get a retriever for the vector database"""
        try:
            client = self.get_client()
            embeddings = self.get_embeddings()
            
            # Check if collection exists
            collections = self.list_collections()
            if self.collection_name not in collections:
                print(f"Collection {self.collection_name} does not exist. Rebuilding database...")
                result = self.rebuild_database()
                if result["status"] != "success":
                    raise Exception(result["message"])
                    
            # Load vector store
            vectordb = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
            
            # Return retriever
            return vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.5,
                    "score_threshold": 0.3
                }
            )
            
        except Exception as e:
            print(f"Error getting retriever: {str(e)}")
            return None