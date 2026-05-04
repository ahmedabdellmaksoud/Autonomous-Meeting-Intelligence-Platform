"""
CorpBrain — Embedder
=====================
Chunks meeting transcripts, embeds them with sentence-transformers,
and stores them in a persistent ChromaDB collection.
Also used by the agentic service to check for duplicate tasks.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

print("[embedder] Loading embedding model...")
_embedder = SentenceTransformer(EMBED_MODEL)
_client   = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Use cosine distance: 0.0 = identical, 1.0 = completely different
# Delete stale collection if it was created with wrong metric
try:
    existing = _client.get_collection(name=COLLECTION_NAME)
    meta     = existing.metadata or {}
    if meta.get("hnsw:space") != "cosine":
        _client.delete_collection(name=COLLECTION_NAME)
        print("[embedder] Recreating collection with cosine metric...")
except Exception:
    pass

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"[embedder] Ready — collection '{COLLECTION_NAME}' has {_collection.count()} chunks.")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def store_transcript(meeting_id: str, transcript: str, metadata: dict | None = None) -> int:
    """
    Chunk, embed, and store a meeting transcript in ChromaDB.

    Args:
        meeting_id: unique meeting identifier
        transcript: full text transcript
        metadata:   optional extra metadata (e.g. date, project)

    Returns:
        Number of chunks stored.
    """
    meta_base = metadata or {}
    chunks    = chunk_text(transcript)

    for i, chunk in enumerate(chunks):
        embedding = _embedder.encode(chunk).tolist()
        _collection.add(
            ids=[f"{meeting_id}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"meeting_id": meeting_id, "chunk_index": i, **meta_base}],
        )

    print(f"[embedder] Stored {len(chunks)} chunks for meeting '{meeting_id}'.")
    return len(chunks)


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search over all stored transcripts.

    Returns list of dicts: [{text, metadata, distance}]
    """
    embedding = _embedder.encode(query).tolist()
    results   = _collection.query(query_embeddings=[embedding], n_results=top_k)

    return [
        {
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
