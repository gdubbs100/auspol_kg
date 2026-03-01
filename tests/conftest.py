import pytest

from auspol_kg.models import Entity, KnowledgeGraph, Relation


@pytest.fixture
def sample_html() -> str:
    return """
    <html><body>
    <script>var x = 1;</script>
    <nav>Navigation</nav>
    <article>
    <h1>Airport Rail Works Start</h1>
    <p>Minister Catherine King today announced $5 billion in funding
    for the Sunshine station as part of the Melbourne Airport Rail project.</p>
    <p>The Victorian Government and the Australian Government are jointly
    funding the project.</p>
    </article>
    <footer>Footer</footer>
    </body></html>
    """


@pytest.fixture
def sample_text() -> str:
    return (
        "Minister Catherine King today announced $5 billion in funding "
        "for the Sunshine station as part of the Melbourne Airport Rail project. "
        "The Victorian Government and the Australian Government are jointly "
        "funding the project."
    )


@pytest.fixture
def sample_kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        entities=[
            Entity(name="Catherine King", entity_type="Person"),
            Entity(name="Melbourne", entity_type="Location"),
            Entity(name="$5 billion", entity_type="Money"),
        ],
        relations=[
            Relation(
                source="Catherine King",
                target="Melbourne",
                relation_type="announce",
            ),
        ],
    )
