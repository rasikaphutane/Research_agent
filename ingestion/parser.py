import fitz
import re
from pathlib import Path
from dataclasses import dataclass, field

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

# ─────────────────────────────────────────────
# Lazy-loaded HuggingFace models
# Only downloaded + initialized on first use
# ─────────────────────────────────────────────

_summarizer = None
_extractor = None


def get_summarizer():
    global _summarizer
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is not installed")
    if _summarizer is None:
        print("  [Loading distilbart ~300MB, one-time...]")
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1,  # CPU
        )
    return _summarizer


def get_extractor():
    global _extractor
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is not installed")
    if _extractor is None:
        print("  [Loading flan-t5-base ~250MB, one-time...]")
        _extractor = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1,
        )
    return _extractor


# ─────────────────────────────────────────────
# HuggingFace extraction functions
# ─────────────────────────────────────────────

def hf_extract_header(first_page: str) -> tuple[str, str]:
    """
    Extract title and authors using flan-t5-base.
    Two focused prompts, one per field.
    Expected: ~2s total on CPU after model is loaded.
    """
    try:
        extractor = get_extractor()

        title_result = extractor(
            f"Extract only the paper title from this text. "
            f"Return just the title, nothing else.\n\n{first_page[:800]}",
            max_new_tokens=40,
            do_sample=False,
        )
        title = title_result[0]["generated_text"].strip()

        author_result = extractor(
            f"Extract only the author names from this academic paper. "
            f"Return full names separated by commas, no affiliations or numbers.\n\n{first_page[:800]}",
            max_new_tokens=60,
            do_sample=False,
        )
        authors = author_result[0]["generated_text"].strip()

        return title, authors

    except Exception as e:
        print(f"  [flan-t5 extraction failed: {e}]")
        return "", ""


def hf_summarize(abstract: str) -> str:
    """
    Summarize abstract using distilbart-cnn-12-6.
    Purpose-built for summarization — faster and more focused than Mistral.
    Expected: ~3s on CPU after model is loaded.
    """
    if not abstract:
        return ""
    try:
        summarizer = get_summarizer()
        result = summarizer(
            abstract[:800],
            max_length=60,
            min_length=20,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"  [distilbart summarization failed: {e}]")
        return ""


# ─────────────────────────────────────────────
# Heuristic fallbacks (zero latency, no models)
# ─────────────────────────────────────────────

def extract_abstract_from_text(text: str) -> str:
    """
    Direct regex extraction — no model needed.
    Abstract always starts at the word 'Abstract' and ends
    at the first section header (Introduction, Keywords, etc.)
    """
    header_pattern = (
        r"(?im)^\s*(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)"
        r"\s*(?:[:.\-—–|])?\s*"
    )
    match = re.search(header_pattern, text)
    if not match:
        match = re.search(r"\bAbstract\b\s*(?:[:.\-—–|])?\s*", text, re.IGNORECASE)
        if not match:
            return ""

    after = text[match.end():]
    end_patterns = [
        r"(?im)^\s*(?:1|I)\s*[\.\-:]?\s+Introduction\b",
        r"(?im)^\s*Introduction\s*$",
        r"(?im)^\s*(?:Keywords?|Index Terms)\s*[:\-]",
        r"(?im)^\s*(?:CCS Concepts|ACM Reference Format|Categories and Subject Descriptors)\b",
        r"(?im)^\s*(?:2|II)\s*[\.\-:]?\s+\w+",
        r"(?i)\s(?:1|I)\s*[\.\-:]?\s+Introduction\b",
        r"(?i)\s(?:Keywords?|Index Terms)\s*[:\-]",
    ]
    ends = []
    for pattern in end_patterns:
        m = re.search(pattern, after)
        if m:
            ends.append(m.start())

    end = min(ends) if ends else len(after)
    abstract = after[:end].strip()
    abstract = re.sub(r"^[\s:.\-—–|]+", "", abstract)
    abstract = re.sub(r"-\n(\w)", r"\1", abstract)   # fix hyphenated line breaks
    abstract = re.sub(r"\s+", " ", abstract)           # normalize whitespace
    return abstract


AUTHOR_NOISE = re.compile(
    r"@|http|www\.|abstract\b|keywords?\b|corresponding author|"
    r"university|institute|department|school|college|faculty|laborator|"
    r"centre|center|campus|research|google brain|google research|"
    r"\bindia\b|\bchina\b|\busa\b|\buk\b|\bjapan\b|\bcanada\b|\bitaly\b|"
    r"\bfrance\b|\bgermany\b|\bcalifornia\b|\bnew york\b|\btamil nadu\b",
    re.IGNORECASE,
)


def _looks_like_person_name(text: str) -> bool:
    text = text.strip()
    if not text or AUTHOR_NOISE.search(text):
        return False
    if len(text) > 70 or len(text.split()) > 6:
        return False
    if re.search(
        r"\b(?:and|the|for|with|using|based|towards?|all|you|need|are|is|of|in|on)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\b[A-Z][A-Za-z'`-]+\b(?:\s+(?:[A-Z]\.?\s+)?[A-Z][A-Za-z'`-]+\b)+",
            text,
        )
    )


def clean_author_string(raw: str) -> str:
    """
    Normalize author text and remove affiliation/email/location bleed.
    Returns an empty string when the input does not look like author names.
    """
    if not raw:
        return ""

    text = re.sub(r"\s+", " ", raw).strip()
    text = re.sub(r"\b(?:and|&)\b", ",", text)
    text = re.sub(r"[\*\u2020\u2021\u00a7]+", "", text)

    parts = []
    for part in re.split(r"\s*(?:;|,|\n)\s*", text):
        part = re.sub(r"^\d+\s*", "", part).strip()
        part = re.sub(r"\s*\d+(?:,\d+)*\s*$", "", part).strip()
        part = re.sub(
            r"\s*\([^)]*(?:university|institute|department|school|@)[^)]*\)",
            "",
            part,
            flags=re.IGNORECASE,
        ).strip()
        if _looks_like_person_name(part):
            parts.append(part)
            continue

        if AUTHOR_NOISE.search(part):
            continue
        for match in re.finditer(
            r"\b[A-Z][A-Za-z'`-]+\s+(?:[A-Z]\.?\s+)?[A-Z][A-Za-z'`-]+\b",
            part,
        ):
            name = match.group(0).strip()
            if _looks_like_person_name(name):
                parts.append(name)

    deduped = []
    seen = set()
    for name in parts:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)

    return ", ".join(deduped)


def heuristic_title(text: str) -> str:
    """
    Last-resort title extraction.
    Finds the first substantial line that isn't a license header,
    institution name, email, or author line.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    skip = [
        r"@",
        r"^\d+$",
        r"permission|copyright|granted|reproduce|preprint|arxiv|submitted",
        r"university|institute|department|school|centre|center",
        r"^\d{4}$",
    ]

    for i, line in enumerate(lines[:20]):
        if len(line) < 20 or line.endswith(","):
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in skip):
            continue

        # Check if next line is a title continuation
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if (
                line[-1] not in ".!?:"
                and len(nxt) < 80
                and not re.search(r"\d|@|university|institute", nxt, re.IGNORECASE)
            ):
                return line + " " + nxt

        return line

    return lines[0] if lines else "Unknown Title"


def _legacy_heuristic_authors(text: str) -> str:
    """
    Last-resort author extraction.
    Searches between title and abstract for name-like lines.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    abs_idx = next(
        (i for i, l in enumerate(lines) if l.lower().startswith("abstract")),
        len(lines)
    )

    found = []
    for line in lines[1:abs_idx]:
        if len(line) > 100:
            continue
        if re.search(
            r"^\d|university|institute|department|school|centre|center|@|http",
            line, re.IGNORECASE
        ):
            continue
        cleaned = re.sub(r"\d+|[∗†‡§]", "", line)
        if len(re.findall(r"\b[A-Z][a-z]+\b", cleaned)) >= 2:
            found.append(re.sub(r"\d+|[∗†‡§*,]+$", "", line).strip())

    return "; ".join(found) if found else "Unknown"


def heuristic_authors(text: str, title: str = "") -> str:
    """
    Author extraction that prefers name-like header lines and stops before
    affiliation/email/abstract content bleeds into the author field.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    abs_idx = next(
        (i for i, l in enumerate(lines) if re.match(r"(?i)^\s*abstract\b", l)),
        len(lines),
    )

    title_words = set(re.findall(r"[a-z]{4,}", title.lower()))
    found = []
    for line in lines[:abs_idx]:
        if len(line) > 160:
            continue
        line_words = set(re.findall(r"[a-z]{4,}", line.lower()))
        if title_words and len(title_words & line_words) >= 3:
            continue
        if AUTHOR_NOISE.search(line):
            if found:
                break
            continue

        cleaned = clean_author_string(line)
        if cleaned:
            found.append(cleaned)
            continue

        no_digits = re.sub(r"\d+", "", line)
        no_digits = re.sub(r"\s+", " ", no_digits).strip()
        cleaned = clean_author_string(no_digits)
        if cleaned:
            found.append(cleaned)

    return clean_author_string(", ".join(found)) or "Unknown"


# ─────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────

@dataclass
class ParsedPaper:
    title: str
    text: str
    metadata: dict
    page_texts: list = field(default_factory=list)
    abstract: str = ""
    summary: str = ""


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def parse_pdf(pdf_path: str, summarize: bool = False) -> ParsedPaper:
    """
    Parse a PDF into a ParsedPaper object.

    Extraction priority (fastest to slowest):
      1. PDF metadata fields  — instant, works for publisher PDFs
      2. HF flan-t5-base      — ~2s, handles messy author-submitted PDFs
      3. Heuristic regex      — instant fallback if models fail

    Summarization (optional):
      distilbart-cnn-12-6     — ~3s, purpose-built summarization model

    Args:
        pdf_path  : path to PDF file
        summarize : generate a one-sentence summary of the abstract.
                    Adds ~3s per paper. Default False for bulk ingestion.
    """
    path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    page_texts = []
    full_text = ""
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            page_texts.append(t)
            full_text += t + "\n\n"

    raw_meta = doc.metadata
    page_count = len(doc)
    doc.close()

    # Use first two pages — some papers spill authors onto page 2
    first_pages = "\n".join(page_texts[:2]) if page_texts else ""

    # ── Step 1: PDF metadata (instant)
    title = raw_meta.get("title", "").strip()
    raw_author = raw_meta.get("author", "").strip()
    author = clean_author_string(raw_author)

    # ── Step 2: Abstract via regex (instant, no model)
    abstract = extract_abstract_from_text(first_pages)

    # ── Step 3: HF extraction if metadata missing
    if not title or not author:
        print("  [extracting header with flan-t5...]")
        hf_title, hf_authors = hf_extract_header(first_pages)

        if not title:
            title = hf_title or heuristic_title(first_pages)
        if not author:
            author = clean_author_string(hf_authors)
            if not author:
                author = heuristic_authors(first_pages, title)

    # ── Step 4: Summary via distilbart (only if requested)
    summary = ""
    if summarize and abstract:
        print("  [summarizing with distilbart...]")
        summary = hf_summarize(abstract)

    metadata = {
        "filename": path.name,
        "title": title,
        "author": author,
        "page_count": page_count,
        "source": str(path.resolve()),
        "abstract": abstract,
        "summary": summary,
    }

    return ParsedPaper(
        title=title,
        text=full_text.strip(),
        metadata=metadata,
        page_texts=page_texts,
        abstract=abstract,
        summary=summary,
    )
