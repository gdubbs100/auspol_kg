import re
from itertools import combinations

import spacy
from spacy.tokens import Span, Token

from ..models import Entity, KnowledgeGraph, Relation
from ..prompts import SPACY_LABEL_TO_TYPE

RELEVANT_TYPES: set[str] = set(SPACY_LABEL_TO_TYPE.keys())

_TRAILING_NOISE: set[str] = {
    "skip", "state", "menu", "navigation", "footer", "header", "close",
    "listen", "submit", "search", "media",
}


def _normalize(name: str) -> str:
    """Normalize entity name for dedup key: strip articles, noise, lowercase."""
    text = " ".join(name.split())  # collapse newlines/whitespace
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    words = text.split()
    while words and words[-1].lower() in _TRAILING_NOISE:
        words.pop()
    return " ".join(words).strip().lower()


def _canonical_name(name: str) -> str:
    """Return a cleaned display name (preserves casing)."""
    text = " ".join(name.split())
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    words = text.split()
    while words and words[-1].lower() in _TRAILING_NOISE:
        words.pop()
    return " ".join(words).strip()


def _dedup_entities(
    seen: dict[str, Entity],
) -> tuple[dict[str, Entity], dict[str, str]]:
    """Merge near-duplicate entities via substring matching.

    Returns (deduped_entities, redirect_map) where redirect maps old keys
    to their canonical key.
    """
    keys = sorted(seen.keys(), key=len)
    redirect: dict[str, str] = {}

    for i, short in enumerate(keys):
        if len(short.split()) < 2:  # skip single-word keys to avoid false merges
            continue
        short_singular = re.sub(r"s$", "", short)
        for long in keys[i + 1 :]:
            if long in redirect:
                continue
            long_singular = re.sub(r"s$", "", long)
            if short_singular in long_singular or short in long:
                redirect[long] = short

    result: dict[str, Entity] = {}
    for key in seen:
        canonical = redirect.get(key, key)
        if canonical not in result:
            result[canonical] = seen[canonical]
    return result, redirect


def _find_connecting_verb(ent_a: Span, ent_b: Span) -> str | None:
    """Find the lemmatized verb connecting two entities via dependency tree."""

    def _verb_ancestors(token: Token) -> list[Token]:
        visited: list[Token] = []
        current = token
        while current.head != current:
            current = current.head
            if current.pos_ == "VERB":
                visited.append(current)
        return visited

    a_verbs = _verb_ancestors(ent_a.root)
    b_verbs = _verb_ancestors(ent_b.root)
    b_verb_ids = {t.i for t in b_verbs}

    # Common verbal ancestor
    for token in a_verbs:
        if token.i in b_verb_ids:
            return token.lemma_

    # Nearest verb of either
    if a_verbs:
        return a_verbs[0].lemma_
    if b_verbs:
        return b_verbs[0].lemma_
    return None


def _determine_direction(
    ent_a: Span, ent_b: Span, key_a: str, key_b: str
) -> tuple[str, str]:
    """Return (source_key, target_key) based on dep roles or doc order."""
    a_is_subj = ent_a.root.dep_ in ("nsubj", "nsubjpass")
    b_is_subj = ent_b.root.dep_ in ("nsubj", "nsubjpass")

    if a_is_subj and not b_is_subj:
        return key_a, key_b
    if b_is_subj and not a_is_subj:
        return key_b, key_a
    # Fallback: document order
    return (key_a, key_b) if ent_a.start < ent_b.start else (key_b, key_a)


def extract_spacy(text: str, model_name: str = "en_core_web_sm") -> KnowledgeGraph:
    """Extract entities via spaCy NER; infer relations via dependency parsing."""
    nlp = spacy.load(model_name)
    doc = nlp(text)

    # Collect and deduplicate entities
    seen: dict[str, Entity] = {}
    for ent in doc.ents:
        if ent.label_ not in RELEVANT_TYPES:
            continue
        key = _normalize(ent.text)
        if key and key not in seen:
            seen[key] = Entity(
                name=_canonical_name(ent.text),
                entity_type=SPACY_LABEL_TO_TYPE[ent.label_],
            )

    seen, redirects = _dedup_entities(seen)

    # Extract relations using dependency parsing
    relations: list[Relation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for sent in doc.sents:
        ents_in_sent: list[tuple[str, Span]] = []
        for e in sent.ents:
            if e.label_ not in RELEVANT_TYPES:
                continue
            key = redirects.get(_normalize(e.text), _normalize(e.text))
            if key in seen:
                ents_in_sent.append((key, e))

        for (key_a, a), (key_b, b) in combinations(ents_in_sent, 2):
            if key_a == key_b:
                continue
            pair = tuple(sorted([key_a, key_b]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            verb = _find_connecting_verb(a, b)
            relation_type = verb if verb else "related_to"
            src_key, tgt_key = _determine_direction(a, b, key_a, key_b)

            # Extract the text span covering both entities
            span_start = min(a.start_char, b.start_char) - sent.start_char
            span_end = max(a.end_char, b.end_char) - sent.start_char
            excerpt = sent.text[max(0, span_start):span_end]

            relations.append(
                Relation(
                    source=seen[src_key].name,
                    target=seen[tgt_key].name,
                    relation_type=relation_type,
                    description=excerpt,
                )
            )

    return KnowledgeGraph(entities=list(seen.values()), relations=relations)
