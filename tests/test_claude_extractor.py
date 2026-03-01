import json
from unittest.mock import MagicMock, patch

from auspol_kg.extraction.claude_extractor import _add_additional_properties_false, extract_claude
from auspol_kg.models import KnowledgeGraph


def test_schema_helper_adds_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        },
    }
    result = _add_additional_properties_false(schema)
    assert result["additionalProperties"] is False
    assert result["properties"]["nested"]["additionalProperties"] is False


def test_schema_helper_handles_defs() -> None:
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"val": {"type": "string"}},
            }
        },
    }
    result = _add_additional_properties_false(schema)
    assert result["$defs"]["Item"]["additionalProperties"] is False


def test_schema_helper_handles_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "things": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"n": {"type": "string"}},
                },
            }
        },
    }
    result = _add_additional_properties_false(schema)
    assert schema["properties"]["things"]["items"]["additionalProperties"] is False


@patch("auspol_kg.extraction.claude_extractor.Anthropic")
def test_extract_claude_parses_response(mock_anthropic_cls: MagicMock) -> None:
    fake_kg = {
        "entities": [{"name": "Test", "entity_type": "PERSON", "description": ""}],
        "relations": [],
    }
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(fake_kg))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    mock_anthropic_cls.return_value = mock_client

    kg = extract_claude("some text")

    assert isinstance(kg, KnowledgeGraph)
    assert len(kg.entities) == 1
    assert kg.entities[0].name == "Test"
