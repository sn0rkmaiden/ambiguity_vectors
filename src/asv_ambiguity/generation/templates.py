from __future__ import annotations

from textwrap import dedent


def build_referent_generation_prompt(topic: str) -> str:
    return dedent(
        f"""
        Create one synthetic training example for the concept of useful referent disambiguation.

        Topic domain: {topic}

        Return the example using exactly these tagged sections and nothing else:
        CONTEXT:
        <short scene description with exactly two plausible referents>
        INSTRUCTION:
        <ambiguous user instruction referring to one of the two referents>
        POSITIVE_RESPONSE:
        <one useful clarifying question that directly resolves which referent is meant>
        NEGATIVE_DIRECT_ANSWER:
        <one direct answer or action that unjustifiably guesses>
        NEGATIVE_WRONG_QUESTION:
        <one grammatical question that does not resolve the referent ambiguity>
        CANDIDATE_REFERENTS:
        <referent 1> | <referent 2>

        Rules:
        - Keep the context short.
        - Make the ambiguity realistic.
        - The positive response must ask about the intended referent, not something else.
        - The negative wrong question must be wrong because it asks about the wrong missing information.
        - Use exactly one line of content under each tag.
        - Do not use JSON.
        - Do not include markdown fences, bullets, or commentary.
        """
    ).strip()
