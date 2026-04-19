from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


OBJECT_SWEEPS = [
    {
        "family": "mug_tray",
        "ambiguity_type": "object_identity",
        "levels": [
            {
                "strength": 0,
                "context": "On the counter there are three mugs: a small blue ceramic mug, a large blue travel mug, and a red ceramic mug. A tray is beside them.",
                "instruction": "Put the large blue travel mug on the tray.",
            },
            {
                "strength": 1,
                "context": "On the counter there are three mugs: a small blue ceramic mug, a large blue travel mug, and a red ceramic mug. A tray is beside them.",
                "instruction": "Put the blue mug on the tray.",
            },
            {
                "strength": 2,
                "context": "On the counter there are three mugs: a small blue ceramic mug, a large blue travel mug, and a red ceramic mug. A tray is beside them.",
                "instruction": "Put the ceramic mug on the tray.",
            },
            {
                "strength": 3,
                "context": "On the counter there are three mugs: a small blue ceramic mug, a large blue travel mug, and a red ceramic mug. A tray is beside them.",
                "instruction": "Put the mug on the tray.",
            },
        ],
    },
    {
        "family": "pitcher_buffet",
        "ambiguity_type": "object_identity",
        "levels": [
            {
                "strength": 0,
                "context": "On the service cart there are three pitchers: a clear pitcher of orange juice, a frosted pitcher of apple juice, and a metal pitcher of water.",
                "instruction": "Bring the frosted pitcher of apple juice to the buffet.",
            },
            {
                "strength": 1,
                "context": "On the service cart there are three pitchers: a clear pitcher of orange juice, a frosted pitcher of apple juice, and a metal pitcher of water.",
                "instruction": "Bring the juice pitcher to the buffet.",
            },
            {
                "strength": 2,
                "context": "On the service cart there are three pitchers: a clear pitcher of orange juice, a frosted pitcher of apple juice, and a metal pitcher of water.",
                "instruction": "Bring the pitcher to the buffet.",
            },
            {
                "strength": 3,
                "context": "The buffet still needs one of the pitchers from the service cart, but there are several ready to go.",
                "instruction": "Bring it to the buffet.",
            },
        ],
    },
]


PREFERENCE_SWEEPS = [
    {
        "family": "drink_choice",
        "ambiguity_type": "preference",
        "levels": [
            {
                "strength": 0,
                "context": "Available drinks are sparkling water, still water, and orange juice.",
                "instruction": "Bring me the sparkling water.",
            },
            {
                "strength": 1,
                "context": "Available drinks are sparkling water, still water, and orange juice.",
                "instruction": "Bring me the water.",
            },
            {
                "strength": 2,
                "context": "Available drinks are sparkling water, still water, and orange juice.",
                "instruction": "Bring me something to drink.",
            },
            {
                "strength": 3,
                "context": "There are several drink options on the counter, but none has been chosen yet.",
                "instruction": "Bring me one.",
            },
        ],
    },
    {
        "family": "seat_selection",
        "ambiguity_type": "preference",
        "levels": [
            {
                "strength": 0,
                "context": "The open seats are window seat A1, aisle seat B2, and quiet corner seat C3.",
                "instruction": "Reserve quiet corner seat C3 for me.",
            },
            {
                "strength": 1,
                "context": "The open seats are window seat A1, aisle seat B2, and quiet corner seat C3.",
                "instruction": "Reserve the quiet seat for me.",
            },
            {
                "strength": 2,
                "context": "The open seats are window seat A1, aisle seat B2, and quiet corner seat C3.",
                "instruction": "Reserve a seat for me.",
            },
            {
                "strength": 3,
                "context": "Several seats are available, each with different tradeoffs, but no preference has been stated.",
                "instruction": "Pick one for me.",
            },
        ],
    },
]


DESTINATION_SWEEPS = [
    {
        "family": "storage_cabinet",
        "ambiguity_type": "destination",
        "levels": [
            {
                "strength": 0,
                "context": "Inside the cabinet there is a top shelf, a middle shelf, and a lower drawer.",
                "instruction": "Put the bowl on the top shelf.",
            },
            {
                "strength": 1,
                "context": "Inside the cabinet there is a top shelf, a middle shelf, and a lower drawer.",
                "instruction": "Put the bowl on the shelf.",
            },
            {
                "strength": 2,
                "context": "Inside the cabinet there is a top shelf, a middle shelf, and a lower drawer.",
                "instruction": "Put the bowl in the cabinet.",
            },
            {
                "strength": 3,
                "context": "The bowl needs to be stored somewhere, but no destination has been specified.",
                "instruction": "Put it away.",
            },
        ],
    },
    {
        "family": "document_dropoff",
        "ambiguity_type": "destination",
        "levels": [
            {
                "strength": 0,
                "context": "The office has an inbox tray, a signature folder, and a locked archive drawer.",
                "instruction": "Place the document in the signature folder.",
            },
            {
                "strength": 1,
                "context": "The office has an inbox tray, a signature folder, and a locked archive drawer.",
                "instruction": "Place the document in the folder.",
            },
            {
                "strength": 2,
                "context": "The office has an inbox tray, a signature folder, and a locked archive drawer.",
                "instruction": "Place the document in the office file area.",
            },
            {
                "strength": 3,
                "context": "The document needs to be put somewhere appropriate, but no destination has been given.",
                "instruction": "Put it where it belongs.",
            },
        ],
    },
]


ALL_SWEEPS = OBJECT_SWEEPS + PREFERENCE_SWEEPS + DESTINATION_SWEEPS



def build_prompt_text(context: str, instruction: str) -> str:
    return f"Context:\n{context}\n\nUser query:\n{instruction}"



def build_rows() -> list[dict]:
    rows: list[dict] = []
    for family_idx, family in enumerate(ALL_SWEEPS):
        for level in family["levels"]:
            strength = int(level["strength"])
            rows.append(
                {
                    "example_id": f"sweep_{family_idx:02d}_{strength}",
                    "family": family["family"],
                    "ambiguity_type": family["ambiguity_type"],
                    "ambiguity_strength": strength,
                    "context": level["context"],
                    "instruction": level["instruction"],
                    "prompt_text": build_prompt_text(level["context"], level["instruction"]),
                    "should_clarify": 1 if strength >= 1 else 0,
                }
            )
    return rows



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", default="outputs/generated/controlled_ambiguity_sweeps_v1.jsonl")
    args = parser.parse_args()

    rows = build_rows()
    out_path = Path(args.output_jsonl)
    write_jsonl(out_path, rows)

    summary = {
        "output": str(out_path.resolve()),
        "num_rows": len(rows),
        "families": sorted({row["family"] for row in rows}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
