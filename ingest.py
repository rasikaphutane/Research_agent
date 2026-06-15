from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer

from graph.builder import KnowledgeGraph
from ingestion.chunker import chunk_paper
from ingestion.embedder import COLLECTION_NAME, embed_and_store, get_chroma_client
from ingestion.parser import parse_pdf


def reset_vector_db():
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Reset Chroma collection: {COLLECTION_NAME}")
    except Exception:
        print(f"No existing Chroma collection to reset: {COLLECTION_NAME}")


def ingest_paper(pdf_path: str, model, kg: KnowledgeGraph):
    print(f"\nParsing: {Path(pdf_path).name}")
    paper = parse_pdf(pdf_path, summarize=False)

    print(f"  Title : {paper.title}")
    print(f"  Author: {paper.metadata.get('author', 'Unknown')}")
    print(f"  Abstract chars: {len(paper.abstract)}")
    print(f"  Pages : {paper.metadata['page_count']}")

    chunks = chunk_paper(paper)
    print(f"  Chunks: {len(chunks)}")
    embed_and_store(chunks, model=model)

    kg.add_paper(paper)


if __name__ == "__main__":
    append_mode = "--append" in sys.argv

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    kg = KnowledgeGraph()
    if append_mode:
        kg.load()
    else:
        print("Rebuilding graph and vector DB from current parser output...")
        reset_vector_db()

    papers_dir = Path("data/papers")
    pdfs = sorted(papers_dir.glob("*.pdf"))

    if not pdfs:
        print("No PDFs in data/papers/")
        sys.exit(1)

    for pdf in pdfs:
        ingest_paper(str(pdf), model, kg)

    print("\nLinking graph...")
    kg.link_citations()
    kg.link_shared_concepts()
    kg.save()

    print("\n--- Graph Stats ---")
    stats = kg.get_stats()
    print(f"  Nodes : {stats['nodes']}")
    print(f"  Edges : {stats['edges']}")
    print(f"  Types : {stats['by_type']}")
    print(f"  Relations: {stats['by_relation']}")
