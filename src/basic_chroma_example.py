"""Basic local ChromaDB example.

This script:
1. Creates (or loads) a local Chroma collection stored under ./chroma_db
2. Inserts a sample document with metadata
3. Performs a similarity query

Run:
    python src/basic_chroma_example.py
"""
from __future__ import annotations
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Initialize client with a local persistence directory
client = chromadb.PersistentClient(path="chroma_db")

# Create OpenAI embedding function 
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

openai_embed_fn = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small", 
    api_key=api_key
)

# Create collection with embedding function (ChromaDB will handle embeddings automatically)
COLLECTION_NAME = "sample_docs"

# Delete existing collection if it exists to avoid conflicts
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"[Info] Deleted existing collection '{COLLECTION_NAME}' to avoid conflicts")
except ValueError:
    # Collection doesn't exist, that's fine
    print(f"[Info] Collection '{COLLECTION_NAME}' doesn't exist yet, creating fresh...")

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
    embedding_function=openai_embed_fn
)
print(f"[Info] Created fresh collection '{COLLECTION_NAME}' with OpenAI embeddings")

# Sample data (expanded with varied content)
ids = ["doc-1", "doc-2", "doc-3"]
texts = [
    # Concise definition
    "Chroma is an open-source embedding database that makes it easy to build LLM apps.",
    # Unrelated (control) document for contrast in similarity results
    "In 1912, explorers ventured across the Antarctic ice, relying on sextants, stars, and sheer endurance to map the last uncharted regions of the Earth.",
    # More detailed Chroma-focused technical description
    (
        "Chroma provides a developer-centric vector database focused on rapid prototyping of retrieval-augmented generation systems. "
        "It abstracts embedding storage, metadata filtering, and approximate nearest neighbor search (HNSW) behind a simple collection interface. "
        "With client-side persistence and pluggable embedding functions, teams can iterate on schema, chunking, and prompt strategies without heavy infrastructure."
    ),
]
metadatas = [
    {"source": "intro", "topic": "chroma", "style": "concise"},
    {"source": "history", "topic": "exploration", "style": "narrative"},
    {"source": "tech", "topic": "chroma", "style": "detailed"},
]

# Upsert documents (ChromaDB will handle embeddings automatically)
collection.upsert(
    ids=ids, 
    documents=texts, 
    metadatas=metadatas
)
print(f"Inserted/Updated {len(ids)} documents: {ids}")

# Query similar docs (ChromaDB will handle query embedding automatically)
query = "What is Chroma?"
results = collection.query(
    query_texts=[query],  # Use query_texts, ChromaDB will embed automatically
    n_results=3
)

print("\nQuery:", query)
for i, (doc, meta, id_) in enumerate(
    zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0], results.get("ids", [[]])[0])
):
    print(f"Result {i+1} | id={id_} | score~? (distance internal) | metadata={meta}\n  {doc}")

print("\nDone.")
