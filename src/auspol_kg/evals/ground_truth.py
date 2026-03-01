"""Layer 2: Compare extracted KG against hand-annotated ground truth."""

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from ..models import KnowledgeGraph


@dataclass
class PrecisionRecall:
    precision: float
    recall: float
    f1: float
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "matched": self.matched, "missed": self.missed, "spurious": self.spurious,
        }


@dataclass
class EvalReport:
    entity_scores: PrecisionRecall
    relation_scores: PrecisionRecall

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_scores": self.entity_scores.to_dict(),
            "relation_scores": self.relation_scores.to_dict(),
        }

    def summary(self) -> str:
        lines = [
            "Ground Truth Evaluation:",
            f"  Entities  - P: {self.entity_scores.precision:.2f}  R: {self.entity_scores.recall:.2f}  F1: {self.entity_scores.f1:.2f}",
            f"    Matched:  {', '.join(self.entity_scores.matched) or 'none'}",
            f"    Missed:   {', '.join(self.entity_scores.missed) or 'none'}",
            f"    Spurious: {', '.join(self.entity_scores.spurious) or 'none'}",
            f"  Relations - P: {self.relation_scores.precision:.2f}  R: {self.relation_scores.recall:.2f}  F1: {self.relation_scores.f1:.2f}",
            f"    Matched:  {', '.join(self.relation_scores.matched) or 'none'}",
            f"    Missed:   {', '.join(self.relation_scores.missed) or 'none'}",
            f"    Spurious: {', '.join(self.relation_scores.spurious) or 'none'}",
        ]
        return "\n".join(lines)


def load_ground_truth(path: str | Path) -> KnowledgeGraph:
    """Load a gold-standard KnowledgeGraph from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgeGraph.model_validate(data)


def _fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two entity names are similar enough to be considered a match."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def _find_match(name: str, candidates: list[str], threshold: float = 0.8) -> str | None:
    """Find the best fuzzy match for a name in a list of candidates."""
    best_score = 0.0
    best_match = None
    for c in candidates:
        score = SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if score >= threshold and score > best_score:
            best_score = score
            best_match = c
    return best_match


def entity_precision_recall(
    extracted: KnowledgeGraph, gold: KnowledgeGraph, threshold: float = 0.8
) -> PrecisionRecall:
    """Compute entity precision/recall using fuzzy name matching."""
    gold_names = [e.name for e in gold.entities]
    extracted_names = [e.name for e in extracted.entities]

    matched: list[str] = []
    missed: list[str] = []
    unmatched_gold = list(gold_names)

    for name in extracted_names:
        match = _find_match(name, unmatched_gold, threshold)
        if match:
            matched.append(f"{name} ~ {match}")
            unmatched_gold.remove(match)

    missed = unmatched_gold
    spurious = [
        n for n in extracted_names
        if not _find_match(n, gold_names, threshold)
    ]

    tp = len(matched)
    precision = tp / len(extracted_names) if extracted_names else 0.0
    recall = tp / len(gold_names) if gold_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PrecisionRecall(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        missed=missed,
        spurious=spurious,
    )


def relation_precision_recall(
    extracted: KnowledgeGraph, gold: KnowledgeGraph, threshold: float = 0.8
) -> PrecisionRecall:
    """Compute relation precision/recall using fuzzy entity name matching."""

    def _rel_key(source: str, target: str) -> str:
        return f"{source} -> {target}"

    gold_rels: list[tuple[str, str, str]] = [
        (r.source, r.target, r.relation_type) for r in gold.relations
    ]
    extracted_rels: list[tuple[str, str, str]] = [
        (r.source, r.target, r.relation_type) for r in extracted.relations
    ]

    matched: list[str] = []
    unmatched_gold = list(gold_rels)

    for e_src, e_tgt, e_type in extracted_rels:
        for g_src, g_tgt, g_type in unmatched_gold:
            if (
                _fuzzy_match(e_src, g_src, threshold)
                and _fuzzy_match(e_tgt, g_tgt, threshold)
            ):
                matched.append(f"{e_src} -[{e_type}]-> {e_tgt}")
                unmatched_gold.remove((g_src, g_tgt, g_type))
                break

    missed = [_rel_key(s, t) for s, t, _ in unmatched_gold]
    spurious_count = len(extracted_rels) - len(matched)
    spurious = [
        f"{s} -[{ty}]-> {t}"
        for s, t, ty in extracted_rels
        if not any(
            _fuzzy_match(s, gs, threshold) and _fuzzy_match(t, gt, threshold)
            for gs, gt, _ in gold_rels
        )
    ]

    tp = len(matched)
    precision = tp / len(extracted_rels) if extracted_rels else 0.0
    recall = tp / len(gold_rels) if gold_rels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PrecisionRecall(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        missed=missed,
        spurious=spurious,
    )


def evaluate(
    extracted: KnowledgeGraph, gold: KnowledgeGraph, threshold: float = 0.8
) -> EvalReport:
    """Run full ground truth evaluation."""
    return EvalReport(
        entity_scores=entity_precision_recall(extracted, gold, threshold),
        relation_scores=relation_precision_recall(extracted, gold, threshold),
    )
