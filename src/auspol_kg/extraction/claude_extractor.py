import json
from typing import Any

from anthropic import Anthropic

from ..models import KnowledgeGraph
from ..prompts import KG_SYSTEM_PROMPT


def _add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false (required by Anthropic API)."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        for prop in schema.get("properties", {}).values():
            _add_additional_properties_false(prop)
    if "items" in schema:
        _add_additional_properties_false(schema["items"])
    if "$defs" in schema:
        for defn in schema["$defs"].values():
            _add_additional_properties_false(defn)
    return schema


def extract_claude(
    text: str, model: str = "claude-sonnet-4-5-20250514", system_prompt: str | None = None
) -> KnowledgeGraph:
    """Extract entities and relationships using Claude structured output."""
    prompt = system_prompt or KG_SYSTEM_PROMPT
    client = Anthropic()
    schema = _add_additional_properties_false(KnowledgeGraph.model_json_schema())

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": f"Extract all entities and relationships from this text:\n\n{text}",
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )

    raw = json.loads(response.content[0].text)
    return KnowledgeGraph.model_validate(raw)
