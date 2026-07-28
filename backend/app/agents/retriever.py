from app.rag.vectorstore import retrieve_chunks

async def run_retriever(query: str, subject: str, n_results: int = 5) -> dict:
    chunks = await retrieve_chunks(query, subject, n_results)

    if not chunks:
        return {
            "answer": "No relevant content found. Please upload documents for this subject first.",
            "sources": []
        }

    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source {i+1} — {c['metadata']['filename']}, chunk {c['metadata']['chunk_index']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    sources = [
        {
            "filename": c["metadata"]["filename"],
            "chunk_index": c["metadata"]["chunk_index"],
            "relevance_score": c["score"]
        }
        for c in chunks
    ]

    return {"context": context, "sources": sources}