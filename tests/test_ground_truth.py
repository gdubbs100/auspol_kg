from auspol_kg.evals.ground_truth import (
    _fuzzy_match,
    entity_precision_recall,
    evaluate,
    relation_precision_recall,
)
from auspol_kg.models import Entity, KnowledgeGraph, Relation


def _make_kg(
    entities: list[tuple[str, str]], relations: list[tuple[str, str, str]]
) -> KnowledgeGraph:
    return KnowledgeGraph(
        entities=[Entity(name=n, entity_type=t) for n, t in entities],
        relations=[
            Relation(source=s, target=t, relation_type=rt) for s, t, rt in relations
        ],
    )


def test_fuzzy_match_exact() -> None:
    assert _fuzzy_match("Catherine King", "Catherine King")


def test_fuzzy_match_close() -> None:
    assert _fuzzy_match("Melbourne Airport Rail Link", "Melbourne Airport Rail")


def test_fuzzy_match_different() -> None:
    assert not _fuzzy_match("Catherine King", "Daniel Mulino")


def test_entity_precision_recall_perfect() -> None:
    kg = _make_kg([("A", "Person"), ("B", "Location")], [])
    gold = _make_kg([("A", "Person"), ("B", "Location")], [])
    scores = entity_precision_recall(kg, gold)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0


def test_entity_precision_recall_partial() -> None:
    extracted = _make_kg([("A", "Person"), ("C", "Person")], [])
    gold = _make_kg([("A", "Person"), ("B", "Location")], [])
    scores = entity_precision_recall(extracted, gold)
    assert scores.precision == 0.5
    assert scores.recall == 0.5
    assert len(scores.missed) == 1
    assert len(scores.spurious) == 1


def test_relation_precision_recall_perfect() -> None:
    extracted = _make_kg(
        [("A", "Person"), ("B", "Location")],
        [("A", "B", "funds")],
    )
    gold = _make_kg(
        [("A", "Person"), ("B", "Location")],
        [("A", "B", "funds")],
    )
    scores = relation_precision_recall(extracted, gold)
    assert scores.precision == 1.0
    assert scores.recall == 1.0


def test_evaluate_returns_report() -> None:
    kg = _make_kg([("A", "Person")], [])
    gold = _make_kg([("A", "Person")], [])
    report = evaluate(kg, gold)
    assert "Ground Truth" in report.summary()
    assert report.entity_scores.f1 == 1.0
