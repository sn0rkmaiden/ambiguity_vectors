# Ambiguity Vectors

This repo now supports exactly two experiment tracks:

1. **Policy vector**
   - target concept: *ask a clarifying question instead of answering when the prompt is ambiguous*
   - data format: paired continuations for the same ambiguous prompt
   - goal: find a broad `clarify_under_ambiguity` / `ask_question_vs_answer` direction and later test steering

2. **Subtype vectors**
   - target concepts: specific ambiguity types inside the clarification regime
   - current concepts:
     - `referent_clarification`
     - `preference_clarification`
     - `destination_clarification`
   - data format: Anthropic-style concept-bearing texts with pooled activations
   - goal: test whether subtype directions generalize and can later explain *what kind* of clarification is needed

The repo intentionally no longer keeps older referent-only scaffolding or superseded config names.

---

## Why there are two tracks

The broad policy vector and subtype vectors answer different questions.

### Policy vector
This is the main intervention target.
It should fire when the model ought to ask a clarifying question rather than guess.
This vector is the best candidate for later steering experiments.

### Subtype vectors
These are analysis vectors.
They should distinguish *why* a prompt needs clarification:
- which referent is meant,
- which preference is intended,
- or which destination / placement is intended.

These are closer to Anthropic's concept-family setup.

---

## Current recommended workflow

### Track A — policy vector
1. Generate paired policy data from realistic ambiguity seeds.
2. Collect activations for labeled responses.
3. Extract a broad vector from positive clarifying responses minus direct-answer negatives.
4. Validate on held-out paired examples.
5. Later: probe it on CLAMBER prompts and run steering tests.

### Track B — subtype vectors
1. Generate concept-bearing texts for each ambiguity subtype.
2. Collect Anthropic-style pooled activations across the generated text span.
3. Extract one vector per subtype.
4. Validate on held-out subtype texts.
5. Later: run controlled ambiguity sweeps and token-level visualization.

---

## Important methodological note

For the subtype track, vectors are **constructed** Anthropic-style by averaging hidden states across token positions in concept-bearing texts after a burn-in offset.

For later diagnostics, the same vectors can still be **probed** at more local positions, including near the prompt/response boundary. These are different choices:
- vector construction
- vector probing / evaluation

This matches the distinction we want for ambiguity work:
- build reusable concept directions from pooled texts,
- then inspect where they activate most strongly in real prompts and responses.

---

## Main files

### Policy track
- `configs/data/policy_pairs_v1.yaml`
- `configs/extraction/policy_positions_v1.yaml`
- `configs/vectors/policy_vector_v1.yaml`
- `src/asv_ambiguity/runners/generate_policy_pairs.py`
- `src/asv_ambiguity/runners/collect_policy_activations.py`
- `src/asv_ambiguity/runners/extract_policy_vector.py`
- `src/asv_ambiguity/runners/validate_policy_vector.py`

### Subtype track
- `configs/data/subtype_concepts_v1.yaml`
- `configs/extraction/concept_pooled_v1.yaml`
- `configs/vectors/subtype_vectors_v1.yaml`
- `src/asv_ambiguity/runners/generate_concept_corpus.py`
- `src/asv_ambiguity/runners/collect_concept_pooled_activations.py`
- `src/asv_ambiguity/runners/extract_concept_vectors.py`
- `src/asv_ambiguity/runners/validate_concept_vectors.py`

### Shared diagnostics
- `src/asv_ambiguity/runners/visualize_vector_activations.py`
- `src/asv_ambiguity/runners/sweep_concept_vectors_on_corpus.py`

---

## Suggested immediate research sequence

1. Build the broad policy vector first.
2. Run held-out validation for layers and positions.
3. Build subtype vectors next.
4. Add controlled ambiguity sweeps.
5. Evaluate on CLAMBER.
6. Only then start steering.

---

## Planned visualizations

These should become standard outputs for the project:

1. **Token heatmaps**
   - token-wise `h_t · v` visualizations on prompts and responses
   - check whether the vector fires on ambiguity-bearing spans or just punctuation / boilerplate

2. **Controlled ambiguity sweeps**
   - vary one missing detail at a time
   - measure how vector activation rises and falls as ambiguity is introduced or resolved

3. **Steering curves**
   - clarification rate on ambiguous prompts versus over-clarification on unambiguous prompts

---

## Repo cleanup policy

This repo is meant to stay small and experiment-friendly.
When a path is clearly superseded, prefer deleting it rather than keeping multiple nearly-identical script families around.

## New evaluation utilities

### CLAMBER prompt-side probing
These scripts support the first external evaluation step for the broad policy vector.

1. Prepare the official CLAMBER dataset:
   `python -m asv_ambiguity.runners.prepare_clamber --output-jsonl outputs/external/clamber_official_v1.jsonl`
2. Probe a policy vector on CLAMBER prompts:
   `python -m asv_ambiguity.runners.evaluate_policy_vector_on_clamber --model-config configs/model/llama32_1b_instruct.yaml --dataset outputs/external/clamber_official_v1.jsonl --vector <VECTOR_PT> --metadata <VECTOR_JSON> --position last_token`

This gives you a prompt-only score for whether the vector activates more strongly on prompts that require clarification.

### Controlled ambiguity sweeps
These scripts generate a small deterministic prompt set where ambiguity is gradually introduced or resolved.

1. Generate the sweep dataset:
   `python -m asv_ambiguity.runners.generate_controlled_ambiguity_sweeps --output-jsonl outputs/generated/controlled_ambiguity_sweeps_v1.jsonl`
2. Score a vector on those sweeps:
   `python -m asv_ambiguity.runners.evaluate_vector_on_controlled_sweeps --model-config configs/model/llama32_1b_instruct.yaml --dataset outputs/generated/controlled_ambiguity_sweeps_v1.jsonl --vector <VECTOR_PT> --metadata <VECTOR_JSON> --position last_token`

Use this to check whether a broad policy vector or subtype vector rises smoothly with ambiguity strength, which is your project’s closest analogue to Anthropic’s controlled prompt interventions.

