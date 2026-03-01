import json
from unittest.mock import MagicMock, patch

from auspol_kg.extraction.cohere_extractor import extract_cohere
from auspol_kg.models import KnowledgeGraph


@patch("auspol_kg.extraction.cohere_extractor.cohere.ClientV2")
def test_extract_cohere_parses_response(mock_client_cls: MagicMock) -> None:
    fake_kg = {
        "entities": [
            {"name": "Catherine King", "entity_type": "Person", "description": "Federal Minister"},
        ],
        "relations": [
            {
                "source": "Catherine King",
                "target": "Melbourne Airport Rail",
                "relation_type": "announces",
                "description": "Catherine King announced the Melbourne Airport Rail project.",
            },
        ],
    }
    mock_content = MagicMock()
    mock_content.text = json.dumps(fake_kg)
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_response = MagicMock()
    mock_response.message = mock_message

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    kg = extract_cohere("some text")

    assert isinstance(kg, KnowledgeGraph)
    assert len(kg.entities) == 1
    assert kg.entities[0].name == "Catherine King"
    assert len(kg.relations) == 1
    assert kg.relations[0].relation_type == "announces"
    assert "Catherine King" in kg.relations[0].description


@patch("auspol_kg.extraction.cohere_extractor.cohere.ClientV2")
def test_extract_cohere_passes_model(mock_client_cls: MagicMock) -> None:
    fake_kg = {"entities": [], "relations": []}
    mock_content = MagicMock()
    mock_content.text = json.dumps(fake_kg)
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_response = MagicMock()
    mock_response.message = mock_message

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    extract_cohere("text", model="command-r-plus-08-2024")

    call_kwargs = mock_client.chat.call_args[1]
    assert call_kwargs["model"] == "command-r-plus-08-2024"
