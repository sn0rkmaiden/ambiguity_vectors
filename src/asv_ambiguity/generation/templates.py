from __future__ import annotations

from textwrap import dedent


def build_referent_generation_prompt(topic: str) -> str:
    return dedent(
        f"""
        Create one synthetic training example for the concept of useful referent disambiguation.

        Topic domain: {topic}

        Return exactly one JSON object with these fields:
        - context: a short scene description with exactly two plausible referents
        - instruction: an ambiguous user instruction referring to one of the two referents
        - positive_response: one useful clarifying question that directly resolves which referent is meant
        - negative_direct_answer: one direct answer or action that unjustifiably guesses
        - negative_wrong_question: one question that is grammatical but does not resolve the referent ambiguity
        - candidate_referents: an array with the two plausible referents
        - missing_slot_type: always the string "referent"

        Rules:
        - Keep the context short.
        - Make the ambiguity realistic.
        - The positive_response must ask about the intended referent, not something else.
        - The negative_wrong_question must be wrong because it asks about the wrong missing information.
        - Do not include markdown or commentary.
        """
    ).strip()
