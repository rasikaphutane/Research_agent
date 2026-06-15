import re
import spacy
from ingestion.parser import ParsedPaper

# Load once at module level — spacy models are reusable
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Noun phrases we don't want as concept nodes
# These are too generic to be meaningful graph nodes
CONCEPT_STOPWORDS = {
    "paper", "work", "study", "approach", "method", "result", "experiment",
    "section", "figure", "table", "model", "system", "data", "dataset",
    "performance", "accuracy", "value", "number", "set", "use", "way",
    "problem", "task", "case", "analysis", "evaluation", "application",
    "training", "testing", "learning", "network", "architecture", "layer",
    "input", "output", "feature", "baseline", "sota", "benchmark",
}


def extract_authors(paper: ParsedPaper) -> list[str]:
    """
    Parse author string into individual names.
    Handles comma-separated and semicolon-separated formats.
    Strips affiliation bleed (digits, symbols, country names).
    """
    raw = paper.metadata.get("author", "")
    if not raw or raw == "Unknown":
        return []

    # Split on semicolons or commas
    parts = re.split(r"[;,]", raw)
    authors = []
    for part in parts:
        name = part.strip()
        # Strip trailing digits, symbols, affiliation fragments
        name = re.sub(r"\d+|[∗†‡§*]", "", name).strip()
        # Skip if it looks like an affiliation, not a name
        if re.search(r"university|institute|department|india|china|usa|school|@",
                     name, re.IGNORECASE):
            continue
        # Skip if too short or no capitalized word
        if len(name) < 4:
            continue
        if not re.search(r"\b[A-Z][a-z]+\b", name):
            continue
        authors.append(name)

    return authors


def extract_concepts(paper: ParsedPaper, top_n: int = 15) -> list[str]:
    """
    Extract key noun phrases from abstract + intro (first 3000 chars).
    Uses spaCy noun chunks, filtered by stopwords and length.
    Returns top N by frequency.

    Why abstract + intro only:
    - Most concept-dense sections of any paper
    - Using full text would drown signal in method details
    - 3000 chars ≈ abstract + first two paragraphs on most papers
    """
    sample = paper.text[:3000]
    doc = nlp(sample)

    freq = {}
    for chunk in doc.noun_chunks:
        concept = chunk.text.lower().strip()
        words = concept.split()

        # Filter criteria:
        # - 2 to 4 words (single words too generic, 5+ too specific)
        # - no stopwords
        # - minimum 6 chars total
        # - no digits (filters "figure 3", "table 1" etc.)
        if (
            2 <= len(words) <= 4
            and not any(w in CONCEPT_STOPWORDS for w in words)
            and len(concept) >= 6
            and not re.search(r"\d", concept)
        ):
            freq[concept] = freq.get(concept, 0) + 1

    sorted_concepts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_concepts[:top_n]]


def extract_references(paper: ParsedPaper) -> list[str]:
    """
    Extract reference identifiers from the References section.

    Priority order:
    1. DOIs — most reliable, structured identifier
    2. Quoted titles — moderately reliable
    3. Numbered reference lines — last resort

    Tradeoff vs Grobid:
    This gives us reference strings, not structured objects.
    Good enough to detect which papers are cited.
    Grobid would give author/year/journal per ref — better for graph
    edges but requires Java server.
    """
    text = paper.text

    ref_match = re.search(
        r"\n(References|Bibliography|Works Cited)\n",
        text,
        re.IGNORECASE
    )
    if not ref_match:
        return []

    ref_text = text[ref_match.end():]

    # 1. DOIs — e.g. 10.1109/CVPR.2016.90
    dois = re.findall(r"10\.\d{4,}/[^\s\]]+", ref_text)

    # 2. Quoted titles — e.g. "Attention is all you need"
    quoted = re.findall(r'"([^"]{15,120})"', ref_text)

    # 3. Numbered lines — e.g. [1] He et al., Deep residual...
    numbered = re.findall(r"\[\d+\]\s+(.{20,150})", ref_text)

    if dois:
        return list(set(dois))[:30]
    if quoted:
        return list(set(quoted))[:30]
    return list(set(numbered))[:20]