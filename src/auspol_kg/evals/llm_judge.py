"""Layer 3: LLM-as-judge evaluation (placeholder).

Intended approach:
- For each entity in the extracted KG, ask an LLM:
  "Given this source text, is '{entity.name}' ({entity.entity_type}) a real
   entity mentioned in the text, or is it hallucinated?"
- For each relation, ask:
  "Given this source text, is the relationship '{source} --{relation_type}--> {target}'
   actually stated or implied? Is the description an accurate quote?"
- Aggregate results into precision scores for grounding.

Not yet implemented. Returns an empty report.
"""

from dataclasses import dataclass


@dataclass
class LLMJudgeReport:
    """Placeholder report for LLM-as-judge evaluation."""
    implemented: bool = False
    message: str = "LLM-as-judge evaluation not yet implemented."

    def summary(self) -> str:
        return self.message


def evaluate_with_llm_judge() -> LLMJudgeReport:
    """Placeholder for LLM-as-judge evaluation."""
    return LLMJudgeReport()
