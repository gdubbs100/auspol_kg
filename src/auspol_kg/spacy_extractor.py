from itertools import combinations

import spacy

from .models import Entity, KnowledgeGraph, Relation

RELEVANT_TYPES: set[str] = {
    "PERSON", "ORG", "GPE", "MONEY", "DATE", "EVENT", "LAW", "FAC", "NORP",
}


def _normalize(name: str) -> str:
    return name.strip().lower()


def extract_spacy(text: str, model_name: str = "en_core_web_sm") -> KnowledgeGraph:
    """Extract entities via spaCy NER; infer relations via sentence co-occurrence."""
    nlp = spacy.load(model_name)
    doc = nlp(text)

    # Deduplicate entities
    seen: dict[str, Entity] = {}
    for ent in doc.ents:
        if ent.label_ not in RELEVANT_TYPES:
            continue
        key = _normalize(ent.text)
        if key not in seen:
            seen[key] = Entity(name=ent.text, entity_type=ent.label_)

    # Co-occurrence relations (entities sharing a sentence)
    relations: list[Relation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for sent in doc.sents:
        ents_in_sent = [
            e for e in sent.ents
            if e.label_ in RELEVANT_TYPES and _normalize(e.text) in seen
        ]
        for a, b in combinations(ents_in_sent, 2):
            pair = tuple(sorted([_normalize(a.text), _normalize(b.text)]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                relations.append(
                    Relation(
                        source=seen[_normalize(a.text)].name,
                        target=seen[_normalize(b.text)].name,
                        relation_type="CO_OCCURS_WITH",
                        description=sent.text[:100],
                    )
                )

    return KnowledgeGraph(entities=list(seen.values()), relations=relations)
