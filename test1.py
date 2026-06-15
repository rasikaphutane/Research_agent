# debug_parser.py
import time
from ingestion.parser import parse_pdf
from pathlib import Path

papers = sorted(Path("data/papers").glob("*.pdf"))

total_start = time.time()
for pdf in papers:
    print(f"\n{'='*60}")
    print(f"File: {pdf.name}")
    t = time.time()
    paper = parse_pdf(str(pdf), summarize=True)
    elapsed = time.time() - t
    print(f"Title   : {paper.title}")
    print(f"Authors : {paper.metadata['author']}")
    print(f"Abstract: {paper.abstract[:200]}...")
    if paper.summary:
        print(f"Summary : {paper.summary}")
    print(f"Time    : {elapsed:.1f}s")

print(f"\nTotal: {time.time() - total_start:.1f}s for {len(papers)} papers")