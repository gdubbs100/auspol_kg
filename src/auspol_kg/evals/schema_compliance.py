"""Layer 1: Automated structural checks on extracted knowledge graphs."""

from dataclasses import dataclass, field

from ..models import KnowledgeGraph
from ..prompts import ENTITY_TYPES


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


@dataclass
class EvalReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {"all_passed": self.all_passed, "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        lines = [f"Schema Compliance: {passed}/{total} checks passed"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}")
            for d in c.details:
                lines.append(f"         {d}")
        return "\n".join(lines)


def check_valid_entity_types(kg: KnowledgeGraph) -> CheckResult:
    """Check all entity_types are in the allowed set."""
    allowed = set(ENTITY_TYPES)
    invalid = [
        f"{e.name!r} has type {e.entity_type!r}"
        for e in kg.entities
        if e.entity_type not in allowed
    ]
    return CheckResult(
        name="valid_entity_types",
        passed=len(invalid) == 0,
        details=invalid,
    )


def check_relation_references(kg: KnowledgeGraph) -> CheckResult:
    """Check all relation source/target match an entity name."""
    entity_names = {e.name for e in kg.entities}
    issues: list[str] = []
    for r in kg.relations:
        if r.source not in entity_names:
            issues.append(f"source {r.source!r} not in entities")
        if r.target not in entity_names:
            issues.append(f"target {r.target!r} not in entities")
    return CheckResult(
        name="relation_references",
        passed=len(issues) == 0,
        details=issues,
    )


def check_no_empty_names(kg: KnowledgeGraph) -> CheckResult:
    """Check no entity has an empty or whitespace-only name."""
    empty = [f"entity at index {i}" for i, e in enumerate(kg.entities) if not e.name.strip()]
    return CheckResult(
        name="no_empty_names",
        passed=len(empty) == 0,
        details=empty,
    )


def check_description_grounded(kg: KnowledgeGraph, source_text: str) -> CheckResult:
    """Check relation descriptions are substrings of the source text."""
    source_lower = source_text.lower()
    ungrounded: list[str] = []
    for r in kg.relations:
        if r.description and r.description.lower() not in source_lower:
            ungrounded.append(
                f"{r.source} -> {r.target}: {r.description[:80]!r}..."
            )
    return CheckResult(
        name="description_grounded",
        passed=len(ungrounded) == 0,
        details=ungrounded,
    )


def run_all_checks(kg: KnowledgeGraph, source_text: str = "") -> EvalReport:
    """Run all schema compliance checks."""
    checks = [
        check_valid_entity_types(kg),
        check_relation_references(kg),
        check_no_empty_names(kg),
    ]
    if source_text:
        checks.append(check_description_grounded(kg, source_text))
    return EvalReport(checks=checks)
