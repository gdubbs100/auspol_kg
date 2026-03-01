from auspol_kg.extraction import parse_html


def test_strips_scripts_and_nav(sample_html: str) -> None:
    text = parse_html(sample_html)
    assert "var x = 1" not in text
    assert "Navigation" not in text
    assert "Footer" not in text


def test_extracts_article_content(sample_html: str) -> None:
    text = parse_html(sample_html)
    assert "Catherine King" in text
    assert "$5 billion" in text


def test_handles_plain_text() -> None:
    text = parse_html("Hello world, no HTML here.")
    assert "Hello world" in text


def test_collapses_blank_lines() -> None:
    html = "<p>Line one</p>\n\n\n\n\n<p>Line two</p>"
    text = parse_html(html)
    assert "\n\n\n" not in text
