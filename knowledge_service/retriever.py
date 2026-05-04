"""
CorpBrain — RAG Retriever
==========================
ChromaDB semantic search → extractive answer (no external API).
When your T5 model is trained on Kaggle, swap _extractive_answer for _t5_answer.
"""

from embedder import search_similar


def _extractive_answer(question: str, chunks: list[dict]) -> str:
    """
    Extractive answer: finds the sentence in retrieved chunks most
    relevant to the question using word overlap scoring.
    No model or API needed.
    """
    if not chunks:
        return "No relevant meeting transcripts found in the knowledge base."

    q_words = set(question.lower().split())
    best_sentence, best_score, best_meeting = "", 0, "unknown"

    for chunk in chunks:
        meeting_id = chunk["metadata"].get("meeting_id", "unknown")
        for sentence in chunk["text"].split(". "):
            s_words = set(sentence.lower().split())
            score   = len(q_words & s_words)
            if score > best_score:
                best_score, best_sentence, best_meeting = score, sentence.strip(), meeting_id

    if best_sentence:
        return f"[From Meeting {best_meeting}]: {best_sentence}"
    return f"[From Meeting {chunks[0]['metadata'].get('meeting_id', 'unknown')}]: {chunks[0]['text'][:200]}"


def answer(question: str, top_k: int = 3) -> dict:
    """
    RAG pipeline: retrieve → extract answer.

    When your T5 model is trained and downloaded from Kaggle,
    replace _extractive_answer with your model's inference call here.

    Returns:
        {answer: str, sources: [{meeting_id, excerpt, distance}]}
    """
    chunks = search_similar(question, top_k=top_k)
    ans    = _extractive_answer(question, chunks)

    return {
        "answer": ans,
        "sources": [
            {
                "meeting_id": c["metadata"]["meeting_id"],
                "excerpt":    c["text"][:120],
                "distance":   c["distance"],
            }
            for c in chunks
        ],
    }


def check_duplicate(task_description: str, threshold: float = 0.45) -> dict:
    """
    Check if a task already exists in stored meetings using ChromaDB cosine distance.
      - 0.0-0.12  = identical / near-identical
      - 0.12-0.45 = strong rewording → flagged as DUPLICATE
      - 0.45-0.70 = related but distinct task
      - 0.70+     = clearly different topic
    Calibrated on all-MiniLM-L6-v2 with real meeting transcripts.
    """
    results = search_similar(task_description, top_k=1)
    if not results:
        return {"is_duplicate": False, "distance": None}

    hit      = results[0]
    distance = hit["distance"]
    if distance < threshold:
        return {
            "is_duplicate": True,
            "similar_to":   hit["text"][:150],
            "meeting_id":   hit["metadata"]["meeting_id"],
            "distance":     round(distance, 4),
        }
    return {"is_duplicate": False, "distance": round(distance, 4)}
