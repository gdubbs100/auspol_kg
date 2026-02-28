from pathlib import Path

from auspol_kg.graph import build_networkx_graph
from auspol_kg.models import KnowledgeGraph
from auspol_kg.visualize import create_comparison_html, visualize_graph


def test_visualize_creates_html(sample_kg: KnowledgeGraph, tmp_path: Path) -> None:
    G = build_networkx_graph(sample_kg)
    out = visualize_graph(G, "Test Graph", tmp_path / "test.html")
    assert out.exists()
    content = out.read_text()
    assert "vis-network" in content or "vis.js" in content.lower() or "<script" in content


def test_comparison_html_has_iframes(tmp_path: Path) -> None:
    # Create dummy HTML files
    (tmp_path / "a.html").write_text("<html>A</html>")
    (tmp_path / "b.html").write_text("<html>B</html>")

    out = create_comparison_html(
        tmp_path / "a.html", tmp_path / "b.html", tmp_path / "compare.html"
    )
    content = out.read_text()
    assert "a.html" in content
    assert "b.html" in content
    assert "<iframe" in content
