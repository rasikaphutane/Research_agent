import networkx as nx
from graph.builder import KnowledgeGraph


class GraphRetriever:
    """
    Query interface over the KnowledgeGraph.
    All methods return plain dicts/lists — no graph objects
    exposed outside this class.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.g = kg.graph

    def get_paper_info(self, filename: str) -> dict:
        """Full metadata for a paper node."""
        if not self.g.has_node(filename):
            return {}
        return dict(self.g.nodes[filename])

    def get_paper_authors(self, filename: str) -> list[str]:
        """Authors of a given paper."""
        return [
            n for n in self.g.successors(filename)
            if self.g.nodes[n].get("type") == "author"
        ]

    def get_paper_concepts(self, filename: str) -> list[str]:
        """Concepts discussed by a given paper."""
        return [
            n for n in self.g.successors(filename)
            if self.g.nodes[n].get("type") == "concept"
        ]

    def get_papers_by_author(self, author_name: str) -> list[dict]:
        """All papers by a given author (fuzzy name match)."""
        results = []
        author_lower = author_name.lower()

        for node, data in self.g.nodes(data=True):
            if data.get("type") != "author":
                continue
            if author_lower in node.lower():
                # Get papers that link to this author
                papers = [
                    p for p in self.g.predecessors(node)
                    if self.g.nodes[p].get("type") == "paper"
                ]
                for paper in papers:
                    results.append({
                        "filename": paper,
                        **self.g.nodes[paper]
                    })
        return results

    def get_papers_by_concept(self, concept: str) -> list[dict]:
        """All papers discussing a concept (fuzzy match)."""
        results = []
        concept_lower = concept.lower()

        for node, data in self.g.nodes(data=True):
            if data.get("type") != "concept":
                continue
            if concept_lower in node.lower():
                papers = [
                    p for p in self.g.predecessors(node)
                    if self.g.nodes[p].get("type") == "paper"
                ]
                for paper in papers:
                    info = dict(self.g.nodes[paper])
                    info["matched_concept"] = node
                    info["filename"] = paper
                    results.append(info)
        return results

    def get_related_concepts(self, concept: str, depth: int = 1) -> list[str]:
        """
        Concepts related to a given concept via co-occurrence.
        depth=1 means direct neighbors only.
        depth=2 would include neighbors-of-neighbors (can be noisy).
        """
        concept_lower = concept.lower()
        matched_node = next(
            (n for n in self.g.nodes
             if self.g.nodes[n].get("type") == "concept"
             and concept_lower in n.lower()),
            None
        )
        if not matched_node:
            return []

        if depth == 1:
            return [
                n for n in self.g.successors(matched_node)
                if self.g.nodes[n].get("type") == "concept"
            ]

        # BFS for depth > 1
        visited = {matched_node}
        frontier = {matched_node}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.g.successors(node):
                    if (neighbor not in visited
                            and self.g.nodes[neighbor].get("type") == "concept"):
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        visited.discard(matched_node)
        return list(visited)

    def get_citations(self, filename: str) -> list[dict]:
        """Papers cited by a given paper (within-corpus only)."""
        return [
            {"filename": n, **self.g.nodes[n]}
            for n in self.g.successors(filename)
            if self.g.nodes[n].get("type") == "paper"
        ]

    def get_all_papers(self) -> list[dict]:
        """All paper nodes with their metadata."""
        return [
            {"filename": n, **d}
            for n, d in self.g.nodes(data=True)
            if d.get("type") == "paper"
        ]