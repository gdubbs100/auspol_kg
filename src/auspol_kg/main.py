from pathlib import Path

from dotenv import load_dotenv

from .extraction import extract_cohere, extract_spacy, load_file, parse_html
from .graph import build_networkx_graph, create_comparison_html, graph_summary, visualize_graph


def run_pipeline(
    input_file: str = "data/2026-02-25_airport-rail-sunshine.html",
    output_dir: str = "output",
) -> None:
    """Run the full KG extraction and visualization pipeline."""
    load_dotenv()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse
    print("Parsing input...")
    raw = load_file(input_file)
    text = parse_html(raw)
    print(f"Extracted {len(text)} chars of text")

    # spaCy extraction
    print("Running spaCy NER...")
    spacy_kg = extract_spacy(text)
    print(f"spaCy: {len(spacy_kg.entities)} entities, {len(spacy_kg.relations)} relations")

    # Cohere extraction
    print("Running Cohere LLM extraction...")
    cohere_kg = extract_cohere(text)
    print(f"Cohere: {len(cohere_kg.entities)} entities, {len(cohere_kg.relations)} relations")

    # Build graphs
    spacy_graph = build_networkx_graph(spacy_kg)
    cohere_graph = build_networkx_graph(cohere_kg)
    print(f"spaCy graph: {graph_summary(spacy_graph)}")
    print(f"Cohere graph: {graph_summary(cohere_graph)}")

    # Visualize
    print("Generating visualizations...")
    spacy_html = visualize_graph(
        spacy_graph, "spaCy NER Knowledge Graph", output_path / "spacy_kg.html"
    )
    cohere_html = visualize_graph(
        cohere_graph, "Cohere LLM Knowledge Graph", output_path / "cohere_kg.html"
    )
    comparison = create_comparison_html(
        spacy_html, cohere_html, output_path / "comparison.html"
    )

    print(f"Done! Open {comparison} in your browser.")


if __name__ == "__main__":
    run_pipeline()
