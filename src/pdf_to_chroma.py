"""
PDF Document Processor for ChromaDB

Processes PDF documents by parsing specified page ranges, splitting into chunks,
and storing in ChromaDB using the ChromaManager class.
"""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# Langchain imports for efficient document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Local imports
from chroma_manager import ChromaDBManager


class PDFToChromaProcessor:
    """Processes PDF documents and stores them in ChromaDB."""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the PDF processor.
        
        Args:
            persist_directory: ChromaDB persistence directory
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chroma_manager = ChromaDBManager(persist_directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_pdf_to_chroma(self,
                            pdf_path: str,
                            collection_name: str,
                            page_range: Optional[Tuple[int, int]] = None,
                            force_recreate: bool = False) -> int:
        """
        Main method to process PDF and store in ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            collection_name: Name of the ChromaDB collection
            page_range: Tuple of (start_page, end_page) 1-indexed, None for all pages
            force_recreate: Whether to recreate the collection
            
        Returns:
            Number of chunks added to the database
        """
        print(f"[Info] Processing PDF: {pdf_path}")
        
        # Load and parse PDF
        documents = self._load_pdf_pages(pdf_path, page_range)
        
        # Split into chunks
        chunks = self._split_documents_to_chunks(documents)
        
        # Prepare chunks for ChromaDB
        chroma_chunks = self._prepare_chunks_for_chroma(chunks, pdf_path)
        
        # Create collection and add documents
        self.chroma_manager.create_collection(collection_name, force_recreate)
        self.chroma_manager.add_documents(collection_name, chroma_chunks)
        
        print(f"[Success] Processed {len(chroma_chunks)} chunks from {len(documents)} pages")
        return len(chroma_chunks)
    
    def _load_pdf_pages(self, pdf_path: str, page_range: Optional[Tuple[int, int]]) -> List[Document]:
        """Load PDF pages using Langchain PyPDFLoader."""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if page_range:
            start_page, end_page = page_range
            # Convert to 0-indexed and validate range
            start_idx = max(0, start_page - 1)
            end_idx = min(len(documents), end_page)
            documents = documents[start_idx:end_idx]
            print(f"[Info] Loading pages {start_page}-{end_page} ({len(documents)} pages)")
        else:
            print(f"[Info] Loading all pages ({len(documents)} pages)")
            
        return documents
    
    def _split_documents_to_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using Langchain text splitter."""
        print(f"[Info] Splitting {len(documents)} pages into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"[Info] Created {len(chunks)} chunks")
        return chunks
    
    def _prepare_chunks_for_chroma(self, chunks: List[Document], pdf_path: str) -> List[Dict[str, str]]:
        """Prepare chunks for ChromaDB storage."""
        chroma_chunks = []
        pdf_name = Path(pdf_path).name
        
        for i, chunk in enumerate(chunks):
            # Extract page number from metadata
            page_num = chunk.metadata.get('page', 0) + 1  # Convert to 1-indexed
            
            chunk_data = {
                'id': str(uuid.uuid4()),
                'content': chunk.page_content,
                'source': pdf_name,
                'page': str(page_num),
                'chunk_index': i,
                'title': f"{pdf_name} - Page {page_num}",
                'timestamp': time.time()
            }
            chroma_chunks.append(chunk_data)
            
        return chroma_chunks


def main():
    """Example usage of the PDFToChromaProcessor."""
    # Initialize processor
    processor = PDFToChromaProcessor(
        persist_directory="chroma_db",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Example: Process a PDF with specific page range
    pdf_path = "docs/constitution_uk.pdf"
    collection_name = "constitution_uk_docs"
    page_range = (2, 19)  # Pages 1-10, None for all pages
    
    try:
        chunk_count = processor.process_pdf_to_chroma(
            pdf_path=pdf_path,
            collection_name=collection_name,
            page_range=page_range,
            force_recreate=True
        )
        print(f"\n[Success] Added {chunk_count} chunks to collection '{collection_name}'")
        
        # Test search functionality
        results = processor.chroma_manager.search_documents(
            collection_name=collection_name,
            #query="What is the voting age in US?",
            query="What is the regime of the UK?",
            n_results=3
        )
        
        print(f"\n[Info] Search test - Found {len(results['documents'][0])} results")
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            print(f"  Result {i+1}: Page {metadata['page']} - {doc[:300]}...")
            
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("[Info] Please ensure the PDF file exists in the specified path")
    except Exception as e:
        print(f"[Error] An error occurred: {e}")


if __name__ == "__main__":
    main()
