# 021 — Hallucination Mitigation: Notes

## What Is Hallucination?

A model generating text that is fluent and confident but factually wrong or unsupported.

Two types:
- **Intrinsic**: contradicts the provided context
- **Extrinsic**: fabricates claims not in any source

## Why It Happens

```
1. Models learn the style of confident text, not the ability to verify claims.
2. Exposure bias: at inference, errors in generated tokens compound.
3. Softmax overconfidence: peaked distribution even without reliable signal.
4. Knowledge gaps: rare facts are underrepresented → model confabulates.
```

## Self-Consistency

Sample the answer N times. Take the majority vote.

```
Why it works:
  There are many wrong answers.
  There is usually only one right answer.
  Hallucinations are inconsistent across samples; correct answers are not.
```

Agreement score = fraction of samples matching the winner.
Low agreement → model is uncertain, treat answer with suspicion.

## Entropy as Uncertainty

```
H(p) = -Σ p(x) log₂ p(x)     (in bits)

Low entropy  → model is confident (distribution is peaked)
High entropy → model is uncertain (distribution is flat)
```

Sequence uncertainty = mean entropy across all token positions.

Use as a cheap inference-time signal — no extra model needed.

## Faithfulness Scoring

When a context is available, check that the response's claims are supported.

```
Faithfulness = (sentences supported by context) / (total sentences)
```

In production: use an NLI (Natural Language Inference) classifier or a small verifier model.

## Abstention

Combine signals to decide whether to refuse to answer:

```python
if entropy > threshold or faithfulness < threshold:
    return "I don't have reliable information about this."
```

Trade-off: higher thresholds → fewer hallucinations, more abstentions.

## RAG as a Mitigation Strategy

RAG (Module 016) converts the problem:

```
Without RAG: model generates from parametric memory → unverifiable
With RAG:    model conditions on retrieved chunks   → verifiable (faithfulness check)
```

Extrinsic hallucination becomes an intrinsic faithfulness problem — much easier to catch.

## Mitigation Strategies at a Glance

| Strategy | Signal | Cost |
|---|---|---|
| Self-consistency | Agreement across N samples | N× inference |
| Entropy threshold | Token distribution flatness | ~free |
| Faithfulness scoring | Overlap with provided context | One extra pass |
| Abstention | Combined signal | Prompt or fine-tune |
| RAG grounding | Retrieval corpus | Retrieval latency |

## The Calibration Insight

A well-calibrated model's confidence matches its accuracy:
- When it says 90% confident → it's correct ~90% of the time
- LLMs out of the box are often overconfident → calibration fine-tuning helps

## Web Search Grounding

Replaces static RAG with a live index.

```
query → search API → top-k snippets → generate → faithfulness check
```

Key difference from RAG:
- RAG: pre-built, fast, can go stale
- Web search: real-time, higher latency, broader coverage

Use web search when facts may post-date the model's training cutoff.

## Self-Critique

Ask the model to review its own output before serving it.

High-risk signal patterns in a sentence:
- Specific years or dates (`1889`, `in 1903`)
- Specific measurements (`330 metres`, `12 kg`)
- Superlatives (`the largest`, `the only`, `the first`)
- Proper nouns — names of people or places

Risk score = fraction of sentences that contain at least one risky pattern.

No external call needed — cheap to run on every generation.

## Citation Enforcement

Force every factual claim to carry a source marker: `[1]`, `[2]`, ...

Two failure modes to catch:
1. **Uncited sentence** — claim has no `[N]` marker (came from parametric memory)
2. **Unsupported citation** — marker is present but the cited snippet doesn't support the claim

Citation coverage = fraction of sentences that are cited.

## Key Papers

- Self-Consistency: Wang et al., 2022 — "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- RAG: Lewis et al., 2020 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Faithfulness: Maynez et al., 2020 — "On Faithfulness and Factuality in Abstractive Summarization"
