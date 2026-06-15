from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from ingestion.chunker import Chunk
from pathlib import Path


DB_PATH = str(Path(__file__).parent.parent / "db")
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)


def get_or_create_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )


def embed_and_store(chunks: list[Chunk], model: SentenceTransformer = None):
    if model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # ChromaDB needs string IDs
    ids = [
        f"{c.metadata['filename']}_chunk_{c.metadata['chunk_index']}"
        for c in chunks
    ]

    # ChromaDB metadata values must be str/int/float/bool only
    safe_metadata = []
    for c in chunks:
        safe_meta = {k: str(v) for k, v in c.metadata.items()}
        safe_metadata.append(safe_meta)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=safe_metadata,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {DB_PATH}")
    return collection


def search(query: str, n_results: int = 5, model: SentenceTransformer = None):
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode([query])[0]

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results