from auspol_kg.evals.schema_compliance import (
    CheckResult,
    check_description_grounded,
    check_no_empty_names,
    check_relation_references,
    check_valid_entity_types,
    run_all_checks,
)
from auspol_kg.models import Entity, KnowledgeGraph, Relation


def test_valid_entity_types_pass(sample_kg: KnowledgeGraph) -> None:
    result = check_valid_entity_types(sample_kg)
    assert result.passed


def test_valid_entity_types_fail() -> None:
    kg = KnowledgeGraph(
        entities=[Entity(name="X", entity_type="Alien")],
        relations=[],
    )
    result = check_valid_entity_types(kg)
    assert not result.passed
    assert any("Alien" in d for d in result.details)


def test_relation_references_pass(sample_kg: KnowledgeGraph) -> None:
    result = check_relation_references(sample_kg)
    assert result.passed


def test_relation_references_fail() -> None:
    kg = KnowledgeGraph(
        entities=[Entity(name="A", entity_type="Person")],
        relations=[Relation(source="A", target="MISSING", relation_type="x")],
    )
    result = check_relation_references(kg)
    assert not result.passed


def test_no_empty_names_pass(sample_kg: KnowledgeGraph) -> None:
    result = check_no_empty_names(sample_kg)
    assert result.passed


def test_no_empty_names_fail() -> None:
    kg = KnowledgeGraph(
        entities=[Entity(name="  ", entity_type="Person")],
        relations=[],
    )
    result = check_no_empty_names(kg)
    assert not result.passed


def test_description_grounded_pass() -> None:
    text = "Catherine King announced funding."
    kg = KnowledgeGraph(
        entities=[Entity(name="Catherine King", entity_type="Person")],
        relations=[
            Relation(
                source="Catherine King",
                target="funding",
                relation_type="announces",
                description="Catherine King announced funding",
            )
        ],
    )
    result = check_description_grounded(kg, text)
    assert result.passed


def test_description_grounded_fail() -> None:
    kg = KnowledgeGraph(
        entities=[Entity(name="A", entity_type="Person")],
        relations=[
            Relation(
                source="A", target="B", relation_type="x",
                description="This text is not in the source",
            )
        ],
    )
    result = check_description_grounded(kg, "Completely different source text.")
    assert not result.passed


def test_run_all_checks(sample_kg: KnowledgeGraph) -> None:
    report = run_all_checks(sample_kg)
    assert len(report.checks) == 3
    assert "Schema Compliance" in report.summary()
