from graph.builder import KnowledgeGraph
from graph.retriever import GraphRetriever

kg = KnowledgeGraph()
kg.load()
gr = GraphRetriever(kg)

print("=== All Papers ===")
for p in gr.get_all_papers():
    print(f"  {p['filename']} — {p['title']}")

print("\n=== Papers by concept: 'attention' ===")
for p in gr.get_papers_by_concept("attention"):
    print(f"  {p['title']} (matched: {p['matched_concept']})")

print("\n=== Concepts in data_p1.pdf ===")
print(" ", gr.get_paper_concepts("data_p1.pdf"))

print("\n=== Authors in data_p2.pdf ===")
print(" ", gr.get_paper_authors("data_p2.pdf"))

print("\n=== Related concepts to 'imaging' ===")
print(" ", gr.get_related_concepts("imaging")[:10])