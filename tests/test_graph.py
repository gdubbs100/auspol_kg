from auspol_kg.graph import ENTITY_COLORS, build_networkx_graph, graph_summary
from auspol_kg.models import KnowledgeGraph, Entity, Relation


def test_builds_correct_node_count(sample_kg: KnowledgeGraph) -> None:
    G = build_networkx_graph(sample_kg)
    assert G.number_of_nodes() == len(sample_kg.entities)


def test_node_has_color(sample_kg: KnowledgeGraph) -> None:
    G = build_networkx_graph(sample_kg)
    node_data = G.nodes["Catherine King"]
    assert node_data["color"] == ENTITY_COLORS["Person"]


def test_skips_edges_with_missing_nodes() -> None:
    kg = KnowledgeGraph(
        entities=[Entity(name="A", entity_type="Person")],
        relations=[Relation(source="A", target="MISSING", relation_type="X")],
    )
    G = build_networkx_graph(kg)
    assert G.number_of_edges() == 0


def test_graph_summary(sample_kg: KnowledgeGraph) -> None:
    G = build_networkx_graph(sample_kg)
    summary = graph_summary(G)
    assert "Nodes: 3" in summary
    assert "Edges: 1" in summary
