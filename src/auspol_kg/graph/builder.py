import networkx as nx

from ..models import KnowledgeGraph

ENTITY_COLORS: dict[str, str] = {
    "Person": "#FF6B6B",
    "Organisation": "#4ECDC4",
    "Location": "#45B7D1",
    "Money": "#96CEB4",
    "Date": "#FFEAA7",
    "Event": "#DDA0DD",
    "Law": "#98D8C8",
    "Infrastructure": "#F7DC6F",
    "Political Group": "#BB8FCE",
    "Facility": "#F0B27A",
}


def build_networkx_graph(kg: KnowledgeGraph) -> nx.Graph:
    """Convert a KnowledgeGraph into a NetworkX graph with visual attributes."""
    G = nx.Graph()

    for entity in kg.entities:
        G.add_node(
            entity.name,
            entity_type=entity.entity_type,
            title=f"{entity.entity_type}: {entity.description}" if entity.description else entity.entity_type,
            color=ENTITY_COLORS.get(entity.entity_type, "#CCCCCC"),
        )

    node_set = set(G.nodes)
    for rel in kg.relations:
        if rel.source in node_set and rel.target in node_set:
            G.add_edge(
                rel.source,
                rel.target,
                label=rel.relation_type,
                title=rel.description,
            )

    return G


def graph_summary(G: nx.Graph) -> str:
    return f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
