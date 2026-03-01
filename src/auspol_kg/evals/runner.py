"""Experiment runner for benchmarking KG extraction approaches."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..extraction import extract_cohere, extract_spacy, load_file, parse_html
from ..extraction.claude_extractor import extract_claude
from ..models import KnowledgeGraph
from .ground_truth import evaluate as gt_evaluate
from .ground_truth import load_ground_truth
from .schema_compliance import run_all_checks


@dataclass
class ExperimentConfig:
    """Configuration for a single evaluation experiment."""

    name: str
    extractor: str  # "spacy" | "cohere" | "claude"
    input_file: str
    ground_truth_file: str
    model: str | None = None
    prompt: str | None = None


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""

    config: ExperimentConfig
    timestamp: str
    duration_seconds: float
    kg: dict
    schema_report: dict
    ground_truth_report: dict


EXTRACTORS: dict[str, Callable[..., KnowledgeGraph]] = {
    "spacy": extract_spacy,
    "cohere": extract_cohere,
    "claude": extract_claude,
}


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single extraction + evaluation experiment."""
    raw = load_file(config.input_file)
    text = parse_html(raw)

    extractor = EXTRACTORS[config.extractor]
    kwargs: dict[str, str] = {}
    if config.model and config.extractor != "spacy":
        kwargs["model"] = config.model
    if config.prompt and config.extractor != "spacy":
        kwargs["system_prompt"] = config.prompt

    start = time.monotonic()
    kg = extractor(text, **kwargs)
    duration = time.monotonic() - start

    schema_report = run_all_checks(kg, text)
    gold = load_ground_truth(config.ground_truth_file)
    gt_report = gt_evaluate(kg, gold)

    return ExperimentResult(
        config=config,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=round(duration, 3),
        kg=kg.model_dump(),
        schema_report=schema_report.to_dict(),
        ground_truth_report=gt_report.to_dict(),
    )


def run_experiments(configs: list[ExperimentConfig]) -> list[ExperimentResult]:
    """Run multiple experiments sequentially."""
    return [run_experiment(c) for c in configs]


def save_results(
    results: list[ExperimentResult], output_dir: str = "data/eval_results"
) -> list[Path]:
    """Save each result as a timestamped JSON file. Returns list of written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for r in results:
        ts = r.timestamp.replace(":", "-")
        path = out / f"{ts}_{r.config.name}.json"
        path.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")
        paths.append(path)
    return paths
