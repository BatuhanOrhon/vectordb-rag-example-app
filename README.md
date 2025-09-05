# VectorDB RAG Example App

A simple PDF document processing and search system using ChromaDB for semantic search capabilities.

## Overview

This project demonstrates how to:

- Process PDF documents and split them into chunks
- Store document chunks in ChromaDB with metadata
- Perform semantic search on stored documents
- Handle multi-document collections safely

## Main Classes

### `PDFToChromaProcessor`

Handles PDF processing and ChromaDB integration:

- Loads PDF documents using Langchain
- Splits documents into chunks with configurable size/overlap
- Stores processed chunks in ChromaDB collections

### `ChromaDBManager`

Manages ChromaDB operations:

- Creates and manages collections
- Adds documents with metadata
- Performs semantic search queries

### `ConstitutionQuerySystem`

Example implementation for multi-document search:

- Sets up constitution documents (UK & US)
- Provides semantic search functionality
- Demonstrates document processing pipeline

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the constitution query system:

   ```bash
   cd src
   python constitution_query_system.py
   ```

3. Or use individual components:
   ```bash
   cd src
   python pdf_to_chroma.py
   ```

## Project Structure

- `src/pdf_to_chroma.py` - PDF processing and ChromaDB integration
- `src/chroma_manager.py` - ChromaDB collection management
- `src/constitution_query_system.py` - Example multi-document system
- `docs/` - Sample PDF documents
- `chroma_db/` - ChromaDB persistent storage (auto-created)

## Dependencies

- ChromaDB - Vector database for semantic search
- Langchain - Document processing and text splitting
- PyPDF2/pypdf - PDF document loading

## License

MIT
