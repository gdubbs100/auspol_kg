from pathlib import Path

import networkx as nx
from pyvis.network import Network


def visualize_graph(
    G: nx.Graph,
    title: str,
    output_path: str | Path,
) -> Path:
    """Create an interactive HTML visualization of a NetworkX graph."""
    output_path = Path(output_path)
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#333333",
        heading=title,
    )
    net.from_nx(G)
    net.force_atlas_2based()
    net.show_buttons(filter_=["physics"])
    net.save_graph(str(output_path))
    return output_path


def create_comparison_html(
    spacy_html_path: Path,
    claude_html_path: Path,
    output_path: str | Path,
) -> Path:
    """Create a side-by-side comparison page embedding both graphs."""
    output_path = Path(output_path)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AusPol KG Comparison</title>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        .container {{ display: flex; height: 100vh; }}
        .panel {{ flex: 1; border: 1px solid #ccc; }}
        .panel h2 {{ text-align: center; margin: 0; padding: 10px; background: #f5f5f5; }}
        iframe {{ width: 100%; height: calc(100% - 45px); border: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h2>spaCy NER + Co-occurrence</h2>
            <iframe src="{spacy_html_path.name}"></iframe>
        </div>
        <div class="panel">
            <h2>Claude API Extraction</h2>
            <iframe src="{claude_html_path.name}"></iframe>
        </div>
    </div>
</body>
</html>"""
    output_path.write_text(html)
    return output_path
