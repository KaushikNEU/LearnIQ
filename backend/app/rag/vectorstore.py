import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings
from app.rag.embeddings import embed_texts, embed_query

settings = get_settings()

chroma_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False)
)

def get_collection(subject: str):
    # Sanitize subject name for ChromaDB
    safe_name = subject.lower().replace(" ", "_").replace("-", "_")
    return chroma_client.get_or_create_collection(
        name=f"learniq_{safe_name}",
        metadata={"hnsw:space": "cosine"}
    )

async def store_chunks(chunks: list[dict], subject: str):
    collection = get_collection(subject)
    texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(texts)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks]
    )
    return len(chunks)

async def retrieve_chunks(query: str, subject: str, n_results: int = 5) -> list[dict]:
    collection = get_collection(subject)
    query_embedding = await embed_query(query)

    # Check collection has documents
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i, doc in enumerate(results["documents"][0]):
        retrieved.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 3)
        })
    return retrieved

def list_subjects() -> list[str]:
    collections = chroma_client.list_collections()
    return [c.name.replace("learniq_", "") for c in collections]

def delete_subject(subject: str):
    safe_name = subject.lower().replace(" ", "_").replace("-", "_")
    chroma_client.delete_collection(f"learniq_{safe_name}")