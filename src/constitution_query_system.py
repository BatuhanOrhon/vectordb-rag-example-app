"""
Multi-Document Query System

Manages multiple constitution documents in ChromaDB and performs safe queries
with source filtering to prevent cross-document contamination.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pdf_to_chroma import PDFToChromaProcessor
from chroma_manager import ChromaDBManager


class ConstitutionQuerySystem:
    """System to manage and query multiple constitution documents safely."""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the multi-document query system.
        
        Args:
            persist_directory: ChromaDB persistence directory
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.processor = PDFToChromaProcessor(persist_directory, chunk_size, chunk_overlap)
        self.chroma_manager = self.processor.chroma_manager
        self.collection_name = "mixed_constitutions"
        
        # Document configurations
        self.documents_config = {
            'uk': {
                'path': 'docs/constitution_uk.pdf',
                'page_range': (2, 19),
                'display_name': 'United Kingdom'
            },
            'us': {
                'path': 'docs/constitution_usa.pdf', 
                'page_range': (2, 18), 
                'display_name': 'United States'
            }
        }
    
    def setup_constitution_database(self) -> Dict[str, int]:
        """
        Insert both UK and US constitution documents into ChromaDB.
        
        Returns:
            Dictionary with chunk counts for each document
        """
        print("🏛️ Setting up Constitution Database")
        print("=" * 40)
        
        results = {}
        
        # Insert UK Constitution first (with collection recreation)
        uk_config = self.documents_config['uk']
        print(f"\nProcessing {uk_config['display_name']} Constitution...")
        
        try:
            uk_chunks = self.processor.process_pdf_to_chroma(
                pdf_path=uk_config['path'],
                collection_name=self.collection_name,
                page_range=uk_config['page_range'],
                force_recreate=True  # Create new collection
            )
            results['uk'] = uk_chunks
        except FileNotFoundError:
            print(f"❌ UK Constitution file not found: {uk_config['path']}")
            results['uk'] = 0
        except Exception as e:
            print(f"❌ Error processing UK Constitution: {e}")
            results['uk'] = 0
        
        # Insert US Constitution (add to existing collection)
        us_config = self.documents_config['us']
        print(f"\nProcessing {us_config['display_name']} Constitution...")
        
        try:
            us_chunks = self.processor.process_pdf_to_chroma(
                pdf_path=us_config['path'],
                collection_name=self.collection_name,
                page_range=us_config['page_range'],
                force_recreate=False  # Add to existing collection
            )
            results['us'] = us_chunks
        except FileNotFoundError:
            print(f"❌ US Constitution file not found: {us_config['path']}")
            results['us'] = 0
        except Exception as e:
            print(f"❌ Error processing US Constitution: {e}")
            results['us'] = 0
        
        # Summary
        total_chunks = sum(results.values())
        print(f"\nDatabase Setup Complete:")
        print(f"   UK Constitution: {results['uk']} chunks")
        print(f"   US Constitution: {results['us']} chunks")
        print(f"   Total: {total_chunks} chunks")
        
        return results
    
    def search(self, query: str, n_results: int = 3) -> Dict:
        """
        Search constitution documents using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        print(f"\nSearching: '{query}'")
        results = self.chroma_manager.search_documents(
            collection_name=self.collection_name,
            query=query,
            n_results=n_results
        )
        
        # Display results with scores
        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return results
        
        print(f"Found {len(results['documents'][0])} results:")
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            source = metadata['source']
            page = metadata['page']
            
            # Get score (distance converted to similarity)
            distance = results.get('distances', [[0]])[0][i] if 'distances' in results else 0
            score = round(1 - distance, 3)  # Convert distance to similarity score
            
            print(f"\n{i+1}. {source} (Page {page}) - Score: {score}")
            print(f"   {doc[:150]}...")
        
        return results



def main():
    """Main function to setup database and perform queries."""
    # Initialize the query system
    query_system = ConstitutionQuerySystem(
        persist_directory="chroma_db",
        chunk_size=500,
        chunk_overlap=50
    )
    
    try:
        # Setup database with both constitutions
        setup_results = query_system.setup_constitution_database()
        
        if setup_results['uk'] == 0 and setup_results['us'] == 0:
            print("❌ No documents were successfully inserted. Please check file paths.")
            return
        
        # Test queries
        test_queries = [
            "What is the age to vote in UK?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing Query: '{query}'")
            print(f"{'='*60}")
            
            # Simple semantic search
            query_system.search(query, n_results=3)
        
        print(f"\nQuery testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
