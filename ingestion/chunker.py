from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.parser import ParsedPaper
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict


def chunk_paper(paper: ParsedPaper, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(paper.text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_meta = {
            **paper.metadata,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
        }
        chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

    return chunks