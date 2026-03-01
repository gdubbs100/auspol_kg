import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from auspol_kg.evals.runner import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    save_results,
)


def _make_config(tmp_path: Path) -> ExperimentConfig:
    html = "<p>Minister Catherine King announced $5 billion in funding.</p>"
    gt = {
        "entities": [
            {"name": "Catherine King", "entity_type": "Person", "description": "Minister"}
        ],
        "relations": [],
    }
    input_file = tmp_path / "input.html"
    input_file.write_text(html)
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(json.dumps(gt))
    return ExperimentConfig(
        name="test-spacy",
        extractor="spacy",
        input_file=str(input_file),
        ground_truth_file=str(gt_file),
    )


def test_run_experiment_spacy(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_experiment(config)

    assert isinstance(result, ExperimentResult)
    assert result.config.name == "test-spacy"
    assert result.duration_seconds >= 0
    assert "entities" in result.kg
    assert "relations" in result.kg
    assert "all_passed" in result.schema_report
    assert "entity_scores" in result.ground_truth_report


def test_save_results(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_experiment(config)

    out_dir = tmp_path / "results"
    paths = save_results([result], str(out_dir))

    assert len(paths) == 1
    assert paths[0].exists()
    data = json.loads(paths[0].read_text())
    assert data["config"]["name"] == "test-spacy"
    assert "entity_scores" in data["ground_truth_report"]


def test_custom_prompt_passed(tmp_path: Path) -> None:
    from auspol_kg.evals import runner
    from auspol_kg.models import KnowledgeGraph

    mock_extract = MagicMock(return_value=KnowledgeGraph(entities=[], relations=[]))
    original = runner.EXTRACTORS["cohere"]
    runner.EXTRACTORS["cohere"] = mock_extract
    try:
        config = _make_config(tmp_path)
        config.extractor = "cohere"
        config.name = "cohere-custom"
        config.prompt = "You are a custom prompt."

        run_experiment(config)

        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a custom prompt."
    finally:
        runner.EXTRACTORS["cohere"] = original
