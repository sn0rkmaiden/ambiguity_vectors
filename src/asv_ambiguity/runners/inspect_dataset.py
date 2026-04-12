from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from asv_ambiguity.utils.io import read_jsonl


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def positive_mentions_candidate(row: dict) -> bool:
    question = normalize(row["positive_response"])
    candidates = [normalize(x) for x in row["candidate_referents"]]
    return any(candidate in question for candidate in candidates)


def print_summary(rows: list[dict]) -> None:
    print(f"Total rows: {len(rows)}")

    split_counts = Counter(row["split"] for row in rows)
    print("\nBy split:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    topic_counts = Counter(row["topic"] for row in rows)
    print("\nBy topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")

    pos_is_question = sum(row["positive_response"].strip().endswith("?") for row in rows)
    neg_direct_has_q = sum("?" in row["negative_direct_answer"] for row in rows)
    neg_wrong_is_question = sum(row["negative_wrong_question"].strip().endswith("?") for row in rows)
    has_two_candidates = sum(len(row["candidate_referents"]) == 2 for row in rows)
    mentions_candidate = sum(positive_mentions_candidate(row) for row in rows)

    print("\nSimple quality checks:")
    print(f"  positive_response ends with '?': {pos_is_question}/{len(rows)}")
    print(f"  negative_direct_answer contains '?': {neg_direct_has_q}/{len(rows)}")
    print(f"  negative_wrong_question ends with '?': {neg_wrong_is_question}/{len(rows)}")
    print(f"  candidate_referents has length 2: {has_two_candidates}/{len(rows)}")
    print(f"  positive_response mentions a candidate string: {mentions_candidate}/{len(rows)}")


def print_examples(rows: list[dict], num_examples: int, show_raw: bool) -> None:
    for i, row in enumerate(rows[:num_examples], start=1):
        print("\n" + "=" * 100)
        print(f"Example {i}")
        print(f"ID: {row['example_id']}")
        print(f"Split: {row['split']}")
        print(f"Topic: {row['topic']}")
        print(f"Candidates: {row['candidate_referents']}")
        print(f"Context: {row['context']}")
        print(f"Instruction: {row['instruction']}")
        print(f"Positive: {row['positive_response']}")
        print(f"Negative direct answer: {row['negative_direct_answer']}")
        print(f"Negative wrong question: {row['negative_wrong_question']}")
        if show_raw:
            raw = row.get("metadata", {}).get("raw_model_output")
            if raw is not None:
                print(f"Raw model output: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)

    if args.split is not None:
        rows = [row for row in rows if row["split"] == args.split]
    if args.topic is not None:
        rows = [row for row in rows if row["topic"] == args.topic]

    if not rows:
        raise ValueError("No rows left after filtering.")

    print_summary(rows)

    rng = random.Random(args.seed)
    sampled = rows[:]
    rng.shuffle(sampled)

    print_examples(sampled, min(args.num_examples, len(sampled)), args.show_raw)


if __name__ == "__main__":
    main()