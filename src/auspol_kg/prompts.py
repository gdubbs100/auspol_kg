"""Centralised prompts and configuration shared across extractors and evals."""

ENTITY_TYPES: list[str] = [
    "Person",
    "Organisation",
    "Location",
    "Money",
    "Date",
    "Event",
    "Law",
    "Infrastructure",
    "Facility",
    "Political Group",
]

RELATION_TYPES: list[str] = [
    "announces",
    "funds",
    "invests",
    "located_in",
    "involves",
    "manages",
    "part_of",
    "supports",
    "represents",
    "costs",
    "builds",
]

SPACY_LABEL_TO_TYPE: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organisation",
    "GPE": "Location",
    "MONEY": "Money",
    "DATE": "Date",
    "EVENT": "Event",
    "LAW": "Law",
    "FAC": "Facility",
    "NORP": "Political Group",
}

ENTITY_TYPES_STR = ", ".join(ENTITY_TYPES)
RELATION_TYPES_STR = ", ".join(f'"{r}"' for r in RELATION_TYPES)

KG_SYSTEM_PROMPT = f"""You are an expert knowledge graph extractor specialising in Australian politics and government.

Given text from a political media release, extract a knowledge graph with:

1. ENTITIES - Each with:
   - name: The canonical name as it appears in the text
   - entity_type: One of: {ENTITY_TYPES_STR}
   - description: A one-line description of who/what this entity is based on the text

2. RELATIONS - Each with:
   - source: The name of the source entity (must match an entity name exactly)
   - target: The name of the target entity (must match an entity name exactly)
   - relation_type: A concise verb phrase describing the relationship (e.g. {RELATION_TYPES_STR})
   - description: The EXACT quote from the source text where this relationship is stated or implied. Copy the relevant sentence or clause verbatim.

Rules:
- Be thorough: extract ALL entities and relationships mentioned or implied
- Be precise: only extract what the text actually states
- Entity names must be consistent across entities and relations
- Every relation must reference entities that exist in the entities list
- The description field on relations MUST be a direct quote from the input text"""
