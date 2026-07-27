# 021 — Hallucination Mitigation

## Goal

By the end of this module, you should be able to answer:

* What is hallucination in LLMs and why does it happen?
* What are the two main types of hallucination?
* What is self-consistency and how does it reduce hallucination?
* How does entropy reveal model uncertainty?
* What is faithfulness scoring and how do you measure it?
* What are the main strategies for mitigating hallucination at inference time?

---

# Theory

## 1. What Is Hallucination?

Hallucination is when a model generates text that is **fluent and confident but factually wrong or unsupported**.

```
Prompt:  "Who invented the telephone?"
Output:  "The telephone was invented by Thomas Edison in 1876."
```

The answer sounds authoritative. It is wrong (Alexander Graham Bell).

The problem: the model has no mechanism to distinguish "I know this" from "I'm completing a plausible pattern."

---

## 2. Types of Hallucination

### Intrinsic Hallucination

The model contradicts a provided source.

```
Context:  "Paris is the capital of France."
Output:   "The capital of France is Lyon."
```

The model has a reference and still gets it wrong.

### Extrinsic Hallucination

The model generates claims not present in (and not verifiable from) any provided source.

```
Context:  "Marie Curie won two Nobel Prizes."
Output:   "Marie Curie also won the Pulitzer Prize in 1912."
```

No source contains this — the model invented it.

---

## 3. Why Do LLMs Hallucinate?

### 3.1 Training Distribution Mismatch

Models are trained on text that was written by humans who knew the facts. The model learns the *style* of confident text, not the *ability* to verify claims.

### 3.2 Exposure Bias

During training, the model sees correct tokens at each step. At inference, it conditions on its own (potentially wrong) outputs — errors compound.

### 3.3 Overconfidence in the Decoding Distribution

Softmax with low temperature produces very peaked distributions. The model picks the most probable token even when it has no reliable signal.

```
P("Edison") = 0.72   ← picked confidently
P("Bell")   = 0.21
P("Meucci") = 0.07
```

### 3.4 Knowledge Gaps

The training corpus doesn't cover everything uniformly. Rare facts are underrepresented → the model fills gaps with plausible-sounding confabulation.

---

## 4. Mitigation Strategy 1 — Self-Consistency

**Idea**: sample the same question N times. If the model is confident and correct, answers converge. If it's hallucinating, answers diverge.

```
Q: "Who invented the telephone?"

Sample 1: "Alexander Graham Bell"
Sample 2: "Alexander Graham Bell"
Sample 3: "Thomas Edison"        ← outlier
Sample 4: "Alexander Graham Bell"
Sample 5: "Alexander Graham Bell"

Majority vote → "Alexander Graham Bell"
```

Self-consistency works because hallucinations tend to be inconsistent across samples — there are many wrong answers but usually one right one.

Originally introduced in chain-of-thought reasoning (Wang et al., 2022).

---

## 5. Mitigation Strategy 2 — Uncertainty Estimation via Entropy

At each token, the model produces a probability distribution over the vocabulary. **Entropy** measures how spread out that distribution is.

```
Low entropy (confident):
  P("Bell") = 0.91, P("Edison") = 0.06, rest ≈ 0
  Entropy ≈ 0.4 bits

High entropy (uncertain):
  P("Bell") = 0.34, P("Edison") = 0.31, P("Meucci") = 0.22, ...
  Entropy ≈ 2.1 bits
```

High entropy at a factual position is a signal that the model is guessing.

**Sequence-level entropy**: average token entropy over the response. High average entropy → flag response for review or abstention.

---

## 6. Mitigation Strategy 3 — Faithfulness Scoring

When a context (document, RAG chunks) is provided, you can score whether the model's output is **faithful** to that context.

A simple faithfulness check:
1. Extract atomic claims from the response.
2. For each claim, ask: "Is this claim supported by the context?"
3. Faithfulness = (supported claims) / (total claims)

```
Context: "The Eiffel Tower is 330 metres tall and located in Paris."

Response: "The Eiffel Tower is 330 metres tall, located in Paris,
           and was built in 1850."   ← last claim not in context

Faithfulness score = 2/3 ≈ 0.67
```

In production this is done with a smaller verifier model or NLI (Natural Language Inference) classifier.

---

## 7. Mitigation Strategy 4 — Abstention

The model can be trained or prompted to say "I don't know" when it is uncertain.

```
"If you are not certain of the answer, say 'I don't have reliable 
information about this' rather than guessing."
```

Combined with uncertainty estimation: if sequence entropy exceeds a threshold, substitute the response with an abstention message.

---

## 8. Mitigation Strategy 5 — RAG as Grounding

Already covered in Module 016. Key point from a mitigation perspective:

RAG forces the model to generate from a verified context window rather than from parametric memory alone. It transforms extrinsic hallucination into an intrinsic faithfulness problem — which is easier to detect.

```
Without RAG: model recalls from training weights → unchecked
With RAG:    model conditions on retrieved chunks → verifiable
```

---

## 9. Summary of Strategies

| Strategy | When to use | Cost |
|---|---|---|
| Self-consistency | Closed-form QA, reasoning | N× inference cost |
| Entropy thresholding | Any generation | Near zero |
| Faithfulness scoring | RAG / document QA | One extra pass |
| Abstention | Safety-critical | Prompt or fine-tune |
| RAG grounding | Knowledge-intensive tasks | Retrieval latency |

---

# Coding Assignments

## Assignment 1 — Self-Consistency

Given a set of N sampled answers to the same question, implement a majority-vote function that returns the most common answer.

```python
def majority_vote(samples: list[str]) -> str:
    pass
```

Also compute an agreement score (fraction of samples that match the winner).

---

## Assignment 2 — Entropy-Based Uncertainty

Given a probability distribution over tokens (or a list of distributions for a sequence), implement:

```python
def token_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a single token distribution."""
    pass

def sequence_uncertainty(token_probs: list[np.ndarray]) -> float:
    """Mean entropy across all token positions."""
    pass
```

Use this to flag high-uncertainty generations.

---

## Assignment 3 — Faithfulness Scorer

Given a context string and a model response, implement a heuristic faithfulness scorer:

```python
def faithfulness_score(context: str, response: str) -> float:
    """
    Split response into sentences (atomic claims).
    For each sentence, check if key terms appear in context.
    Return fraction of supported sentences.
    """
    pass
```

---

## Assignment 4 — Abstention Threshold

Combine uncertainty and faithfulness into a decision:

```python
def should_abstain(token_probs, context, response,
                   entropy_threshold=1.5,
                   faithfulness_threshold=0.5) -> bool:
    pass
```

Return True if the model should abstain (entropy too high or faithfulness too low).

---

# Success Criteria

* Understand intrinsic vs extrinsic hallucination
* Implement self-consistency majority vote
* Compute token and sequence entropy
* Build a heuristic faithfulness scorer
* Build an abstention decision function combining both signals
