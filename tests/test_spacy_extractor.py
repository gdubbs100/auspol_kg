from auspol_kg.models import KnowledgeGraph
from auspol_kg.spacy_extractor import extract_spacy


def test_returns_knowledge_graph(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    assert isinstance(kg, KnowledgeGraph)


def test_finds_person_entities(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    names = {e.name for e in kg.entities if e.entity_type == "PERSON"}
    assert any("Catherine King" in n for n in names)


def test_finds_org_entities(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    types = {e.entity_type for e in kg.entities}
    # Should find at least one ORG or GPE
    assert types & {"ORG", "GPE"}


def test_deduplicates_entities(sample_text: str) -> None:
    doubled = sample_text + " " + sample_text
    kg = extract_spacy(doubled)
    names = [e.name.lower() for e in kg.entities]
    assert len(names) == len(set(names))


def test_creates_cooccurrence_relations(sample_text: str) -> None:
    kg = extract_spacy(sample_text)
    assert all(r.relation_type == "CO_OCCURS_WITH" for r in kg.relations)
