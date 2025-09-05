"""
ChromaDB Collection Manager

Manages ChromaDB collections for storing document chunks.
"""
from __future__ import annotations
import chromadb
from typing import List, Dict, Optional
import time


class ChromaDBManager:
    """Manages ChromaDB collections for document storage."""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
        
    def create_collection(self, collection_name: str, force_recreate: bool = False) -> chromadb.Collection:
        """
        Create or get a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            force_recreate: If True, delete existing collection and create new one
            
        Returns:
            ChromaDB Collection object
        """
        if force_recreate:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"[Info] Deleted existing collection '{collection_name}'")
            except Exception:
                # Collection doesn't exist
                pass
        
        try:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[Info] Created new collection '{collection_name}'")
        except Exception:
            # Collection already exists
            collection = self.client.get_collection(name=collection_name)
            print(f"[Info] Using existing collection '{collection_name}'")
        
        self.collections[collection_name] = collection
        return collection
    
    def add_documents(self, collection_name: str, chunks: List[Dict[str, str]]) -> None:
        """
        Add document chunks to ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            chunks: List of document chunks with metadata
        """
        collection = self.collections[collection_name]
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk['id'])
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk.get('source', ''),
                'page': chunk.get('page', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'title': chunk.get('title', ''),
                'timestamp': chunk.get('timestamp', time.time())
            })
        
        # Add to collection
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"[Info] Added {len(chunks)} chunks to collection '{collection_name}'")
    
    def search_documents(self, collection_name: str, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar documents in the collection.
        
        Args:
            collection_name: Name of the collection
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results from ChromaDB
        """
        if collection_name not in self.collections:
            collection = self.client.get_collection(name=collection_name)
            self.collections[collection_name] = collection
        
        collection = self.collections[collection_name]
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
