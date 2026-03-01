from auspol_kg.models import KnowledgeGraph
from auspol_kg.extraction import extract_spacy


def test_returns_knowledge_graph(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    assert isinstance(kg, KnowledgeGraph)


def test_finds_person_entities(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    names = {e.name for e in kg.entities if e.entity_type == "Person"}
    assert any("Catherine King" in n for n in names)


def test_finds_org_entities(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    types = {e.entity_type for e in kg.entities}
    assert types & {"Organisation", "Location"}


def test_deduplicates_entities(sample_text: str) -> None:
    doubled = sample_text + " " + sample_text
    kg = extract_spacy(doubled)
    names = [e.name.lower() for e in kg.entities]
    assert len(names) == len(set(names))


def test_creates_verb_relations(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    assert len(kg.relations) > 0
    relation_types = {r.relation_type for r in kg.relations}
    # Should have at least one verb-based relation (not all "related_to")
    assert relation_types != {"related_to"}


def test_normalizes_trailing_noise() -> None:
    text = "The Department of Infrastructure Skip announced new funding."
    kg = extract_spacy(text)
    names = [e.name for e in kg.entities]
    assert not any("Skip" in n for n in names)
