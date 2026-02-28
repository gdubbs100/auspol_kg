from pathlib import Path

from .graph import build_networkx_graph, graph_summary
from .parser import load_file, parse_html
from .spacy_extractor import extract_spacy
from .visualize import visualize_graph


def run_pipeline(
    input_file: str = "test_text.txt",
    output_dir: str = "output",
) -> None:
    """Run the full KG extraction and visualization pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse
    print("Parsing input...")
    raw = load_file(input_file)
    text = parse_html(raw)
    print(f"Extracted {len(text)} chars of text")

    # spaCy extraction
    print("Running spaCy NER...")
    kg = extract_spacy(text)
    print(f"Found {len(kg.entities)} entities, {len(kg.relations)} relations")

    # Build graph
    graph = build_networkx_graph(kg)
    print(f"Graph: {graph_summary(graph)}")

    # Visualize
    print("Generating visualization...")
    out_html = visualize_graph(graph, "AusPol Knowledge Graph", output_path / "kg.html")

    print(f"Done! Open {out_html} in your browser.")


if __name__ == "__main__":
    run_pipeline()
