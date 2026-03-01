import json

import cohere

from ..models import KnowledgeGraph
from ..prompts import KG_SYSTEM_PROMPT


def extract_cohere(
    text: str, model: str = "command-a-03-2025", system_prompt: str | None = None
) -> KnowledgeGraph:
    """Extract entities and relationships using Cohere LLM."""
    prompt = system_prompt or KG_SYSTEM_PROMPT
    client = cohere.ClientV2()

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Extract a complete knowledge graph from this media release:\n\n{text}",
            },
        ],
        response_format={
            "type": "json_object",
            "json_schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "entity_type": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "entity_type", "description"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": [
                                "source",
                                "target",
                                "relation_type",
                                "description",
                            ],
                        },
                    },
                },
                "required": ["entities", "relations"],
            },
        },
    )

    raw = json.loads(response.message.content[0].text)
    return KnowledgeGraph.model_validate(raw)
