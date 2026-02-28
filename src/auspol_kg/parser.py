import re
from pathlib import Path

from bs4 import BeautifulSoup


def load_file(filepath: str | Path) -> str:
    return Path(filepath).read_text(encoding="utf-8")


def parse_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Try to find article content first
    article = (
        soup.find("article")
        or soup.find("div", class_="field-item")
        or soup.find("div", class_="content")
    )
    target = article if article else soup

    text = target.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
