import json
import networkx as nx
from pathlib import Path
from ingestion.parser import ParsedPaper
from graph.extractor import extract_authors, extract_concepts, extract_references

GRAPH_PATH = Path(__file__).parent.parent / "db" / "knowledge_graph.json"


class KnowledgeGraph:
    """
    Directed graph over papers, authors, and concepts.

    Node types:
        paper   — one node per PDF, keyed by filename
        author  — one node per unique author name
        concept — one node per unique noun phrase

    Edge types:
        authored_by  — paper → author
        discusses    — paper → concept
        cites        — paper → paper (within-corpus only)
        related_to   — concept → concept (shared across 2+ papers)
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_paper(self, paper: ParsedPaper):
        filename = paper.metadata["filename"]
        title = paper.metadata.get("title", filename)

        # ── Paper node
        self.graph.add_node(
            filename,
            type="paper",
            title=title,
            author=paper.metadata.get("author", ""),
            abstract=paper.metadata.get("abstract", "")[:300],
            summary=paper.metadata.get("summary", ""),
            page_count=paper.metadata.get("page_count", 0),
        )

        # ── Author nodes + edges
        authors = extract_authors(paper)
        for author in authors:
            if not self.graph.has_node(author):
                self.graph.add_node(author, type="author")
            self.graph.add_edge(filename, author, relation="authored_by")

        # ── Concept nodes + edges
        concepts = extract_concepts(paper)
        for concept in concepts:
            if not self.graph.has_node(concept):
                self.graph.add_node(concept, type="concept")
            self.graph.add_edge(filename, concept, relation="discusses")

        # ── Store raw references for cross-linking later
        refs = extract_references(paper)
        self.graph.nodes[filename]["raw_references"] = refs

        print(f"  Graph: {len(authors)} authors | {len(concepts)} concepts | {len(refs)} refs")
        return authors, concepts, refs

    def link_citations(self):
        """
        After all papers are ingested, try to link citation edges
        within the corpus.

        Strategy: for each paper's raw_references, check if any
        other paper's title appears as a substring. Fuzzy but
        effective for small corpora.

        Tradeoff: false positives possible if titles are short.
        Grobid would give us DOIs for exact matching.
        """
        paper_nodes = [
            (n, d) for n, d in self.graph.nodes(data=True)
            if d.get("type") == "paper"
        ]

        links_added = 0
        for filename, data in paper_nodes:
            refs = data.get("raw_references", [])
            for ref_text in refs:
                for other_file, other_data in paper_nodes:
                    if other_file == filename:
                        continue
                    other_title = other_data.get("title", "").lower()
                    if len(other_title) > 10 and other_title in ref_text.lower():
                        self.graph.add_edge(
                            filename, other_file,
                            relation="cites"
                        )
                        links_added += 1

        print(f"  Citation links added: {links_added}")

    def link_shared_concepts(self):
        """
        Add related_to edges between concepts discussed by 2+ papers.
        Two concepts are related if they co-occur in the same paper.

        This creates a concept co-occurrence network — useful for
        "what topics are related to X" queries.
        """
        concept_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "concept"
        ]

        edges_added = 0
        for concept in concept_nodes:
            # Papers that discuss this concept
            papers = [
                p for p in self.graph.predecessors(concept)
                if self.graph.nodes[p].get("type") == "paper"
            ]
            # Mark how many papers share this concept
            self.graph.nodes[concept]["paper_count"] = len(papers)

            # Co-occurring concepts in same papers
            if len(papers) > 1:
                for paper in papers:
                    for other_concept in self.graph.successors(paper):
                        if (
                            other_concept != concept
                            and self.graph.nodes[other_concept].get("type") == "concept"
                            and not self.graph.has_edge(concept, other_concept)
                        ):
                            self.graph.add_edge(
                                concept, other_concept,
                                relation="related_to"
                            )
                            edges_added += 1

        print(f"  Concept relation edges added: {edges_added}")

    def get_stats(self) -> dict:
        by_type = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        by_relation = {}
        for _, _, d in self.graph.edges(data=True):
            r = d.get("relation", "unknown")
            by_relation[r] = by_relation.get(r, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "by_type": by_type,
            "by_relation": by_relation,
        }

    def save(self):
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Graph saved → {GRAPH_PATH}")

    def load(self) -> bool:
        if not GRAPH_PATH.exists():
            print("  No existing graph found, starting fresh")
            return False
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        print(f"  Graph loaded: {self.get_stats()}")
        return True