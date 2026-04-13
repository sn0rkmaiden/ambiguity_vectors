# Ambiguity Vectors

Scaffold for reproducing the concept-vector pipeline for ambiguity resolution:

1. generate a narrow synthetic concept dataset,
2. replay the generated texts through the same model and collect hidden activations,
3. extract a dense steering vector as a mean activation difference.

The target concept is 
> asking a useful referent-disambiguation question.

## Repo layout

```text
anthropic_style_ambiguity_vectors_starter/
  configs/
    data/referent_disambiguation.yaml
    extraction/first_assistant_token.yaml
    model/llama32_1b_instruct.yaml
    vectors/referent_vector.yaml
  data/
    concepts/referent_disambiguation/topics.json
  outputs/
  src/asv_ambiguity/
```

## Dataset schema

Each generated example contains:
- `context`: short scene or environment
- `instruction`: ambiguous user instruction
- `positive_response`: a useful clarifying question
- `negative_direct_answer`: a direct but unjustified guess
- `negative_wrong_question`: a question that does not resolve the ambiguity
- `missing_slot_type`: what information is missing
- `candidate_referents`: optional list of plausible referents
- `split`: train / val / test

## Install

```bash
pip install -e .
```

## First run

Generate synthetic examples:

```bash
python -m asv_ambiguity.runners.generate_dataset \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --data-config configs/data/clarification_seeded_v1.yaml
```

Inspect generated dataset:

```bash
python -m asv_ambiguity.runners.inspect_dataset \
  --dataset outputs/generated/{paste_filename}.jsonl \
  --num-examples 12 \
  --show-raw
```

Collect activations:

```bash
# 1B model
python -m asv_ambiguity.runners.collect_activations   --model-config configs/model/llama32_1b_instruct.yaml   --extraction-config configs/extraction/first_assistant_token.yaml   --dataset outputs/generated/{paste_filename}.jsonl

# 8B model
python -m asv_ambiguity.runners.collect_activations   --model-config configs/model/llama31_8b_instruct.yaml   --extraction-config configs/extraction/first_assistant_token.yaml   --dataset outputs/generated/{paste_filename}.jsonl

# collect activations on multiple positions
python -m asv_ambiguity.runners.collect_activations \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --extraction-config configs/extraction/multi_positions.yaml \
  --dataset outputs/generated/{paste_filename}.jsonl
```

Extract the first dense vector:

```bash
python -m asv_ambiguity.runners.extract_vector   --vector-config configs/vectors/referent_vector.yaml   --activations outputs/activations/referent_disambiguation_first_assistant_token.pt

# extract vectors from multiple positions (if multi_position.yaml was used in command before)
python -m asv_ambiguity.runners.extract_vector \
  --vector-config configs/vectors/multi_position_vectors.yaml \
  --activations outputs/activations/{paste_activations_filename}.pt
```

Score the vector on held-out rows, compare positive response against the two negatives, check whether the positive gets the highest score more often than chance:
```bash
python -m asv_ambiguity.runners.validate_vector \
  --model-config configs/model/llama31_8b_instruct.yaml \
  --dataset outputs/generated/{paste_filename}.jsonl \
  --vector outputs/vectors/referent_disambiguation_layer12.pt \
  --metadata outputs/vectors/referent_disambiguation_layer12.json \
  --splits val test

# validate all vectors from multiple positions (provided that the necessary activations were collected beforehand)
python -m asv_ambiguity.runners.validate_vector --model-config configs/model/llama31_8b_instruct.yaml --dataset outputs/generated/{paste_filename}.jsonl --vector outputs/vectors/{paste vector file for validation}.pt --metadata outputs/vectors/{paste vector file for validation}.json --splits val test
```

Run visualization:

```bash
export NEURONPEDIA_API_KEY="your_secret_key"

python visualize_with_neuronpedia.py \
  --vector outputs/vectors/clarification_seeded_v1__last_question_token__layer16.pt \
  --metadata outputs/vectors/clarification_seeded_v1__last_question_token__layer16.json \
  --prompt-file example_prompt.txt \
  --output-html outputs/visualizations/neuronpedia_example.html \
  --model-id llama3.1-8b-it
```
