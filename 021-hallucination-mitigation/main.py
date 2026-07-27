import numpy as np
from collections import Counter

np.random.seed(42)


# --------------------------
# Assignment 1 — Self-Consistency
# --------------------------

print("=== Assignment 1: Self-Consistency ===")


def majority_vote(samples: list) -> tuple:
    """
    Return (winner, agreement_score) where:
      winner          = most common answer
      agreement_score = fraction of samples matching the winner
    """
    counts = Counter(s.strip().lower() for s in samples)
    winner_lower, winner_count = counts.most_common(1)[0]
    # Return the original-casing version of the winner
    winner = next(s for s in samples if s.strip().lower() == winner_lower)
    agreement = winner_count / len(samples)
    return winner, agreement


questions = [
    {
        "question": "Who invented the telephone?",
        "samples": [
            "Alexander Graham Bell",
            "Alexander Graham Bell",
            "Thomas Edison",
            "Alexander Graham Bell",
            "Alexander Graham Bell",
        ],
        "correct": "Alexander Graham Bell",
    },
    {
        "question": "What is the capital of Australia?",
        "samples": [
            "Sydney",
            "Canberra",
            "Canberra",
            "Melbourne",
            "Canberra",
        ],
        "correct": "Canberra",
    },
    {
        "question": "What is the boiling point of water in Celsius?",
        "samples": [
            "100 degrees",
            "100 degrees",
            "212 degrees",   # Fahrenheit — hallucinated unit
            "100 degrees",
            "100 degrees",
        ],
        "correct": "100 degrees",
    },
]

for q in questions:
    answer, agreement = majority_vote(q["samples"])
    correct = answer.strip().lower() == q["correct"].strip().lower()
    print(f"\n  Q: {q['question']}")
    print(f"     Majority answer : {answer}")
    print(f"     Agreement score : {agreement:.0%}")
    print(f"     Correct         : {correct}")


# --------------------------
# Assignment 2 — Entropy-Based Uncertainty
# --------------------------

print("\n=== Assignment 2: Entropy-Based Uncertainty ===")


def token_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a single token distribution (in bits)."""
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def sequence_uncertainty(token_probs: list) -> float:
    """Mean entropy across all token positions."""
    return float(np.mean([token_entropy(p) for p in token_probs]))


# Simulate token distributions for two responses.
# Each entry is a short probability distribution over a tiny vocabulary.
# (In practice this would be over 50k+ tokens.)

VOCAB = ["Bell", "Edison", "Meucci", "Watson", "Morse", "Other"]

# Confident, correct response — model peaks sharply on "Bell"
confident_token_probs = [
    np.array([0.91, 0.05, 0.02, 0.01, 0.005, 0.005]),
    np.array([0.95, 0.02, 0.01, 0.01, 0.005, 0.005]),
    np.array([0.89, 0.06, 0.02, 0.01, 0.01,  0.01 ]),
]

# Uncertain, hallucinating response — distribution is flat
uncertain_token_probs = [
    np.array([0.30, 0.28, 0.20, 0.12, 0.05, 0.05]),
    np.array([0.25, 0.30, 0.22, 0.13, 0.05, 0.05]),
    np.array([0.35, 0.25, 0.18, 0.12, 0.05, 0.05]),
]

conf_entropy   = sequence_uncertainty(confident_token_probs)
uncert_entropy = sequence_uncertainty(uncertain_token_probs)

print(f"\n  Confident response  — mean entropy: {conf_entropy:.3f} bits")
print(f"  Uncertain response  — mean entropy: {uncert_entropy:.3f} bits")

ENTROPY_THRESHOLD = 1.0
print(f"\n  Threshold: {ENTROPY_THRESHOLD} bits")
print(f"  Flag confident  : {conf_entropy   > ENTROPY_THRESHOLD}")
print(f"  Flag uncertain  : {uncert_entropy > ENTROPY_THRESHOLD}")

# Show per-token entropy for both
print("\n  Per-token entropy (confident):")
for i, p in enumerate(confident_token_probs):
    print(f"    Token {i+1}: {token_entropy(p):.3f} bits  (peak: {VOCAB[np.argmax(p)]})")

print("  Per-token entropy (uncertain):")
for i, p in enumerate(uncertain_token_probs):
    print(f"    Token {i+1}: {token_entropy(p):.3f} bits  (peak: {VOCAB[np.argmax(p)]})")


# --------------------------
# Assignment 3 — Faithfulness Scorer
# --------------------------

print("\n=== Assignment 3: Faithfulness Scorer ===")


def _tokenize(text: str) -> set:
    """Lowercase word tokens, stripping punctuation."""
    import re
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def faithfulness_score(context: str, response: str) -> float:
    """
    Split response into sentences (atomic claims).
    A sentence is considered 'supported' if the majority of its
    content words (len >= 3) appear in the context.
    Returns the fraction of supported sentences.
    """
    sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if not sentences:
        return 0.0

    context_tokens = _tokenize(context)
    supported = 0

    for sentence in sentences:
        sent_tokens = _tokenize(sentence)
        if not sent_tokens:
            supported += 1
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap >= 0.5:
            supported += 1

    return supported / len(sentences)


examples = [
    {
        "label": "Fully faithful",
        "context": "The Eiffel Tower is 330 metres tall and located in Paris, France. It was built in 1889.",
        "response": "The Eiffel Tower is 330 metres tall. It is located in Paris. It was built in 1889.",
    },
    {
        "label": "Partially faithful (one hallucinated claim)",
        "context": "The Eiffel Tower is 330 metres tall and located in Paris, France. It was built in 1889.",
        "response": "The Eiffel Tower is 330 metres tall. It is located in Paris. It won the UNESCO award in 1920.",
    },
    {
        "label": "Mostly hallucinated",
        "context": "Marie Curie won two Nobel Prizes.",
        "response": "Marie Curie won three Nobel Prizes. She also won the Pulitzer Prize. She was born in Germany.",
    },
]

for ex in examples:
    score = faithfulness_score(ex["context"], ex["response"])
    print(f"\n  [{ex['label']}]")
    print(f"    Context  : {ex['context'][:70]}...")
    print(f"    Response : {ex['response'][:70]}...")
    print(f"    Faithfulness score: {score:.2f}")


# --------------------------
# Assignment 4 — Abstention Threshold
# --------------------------

print("\n=== Assignment 4: Abstention Decision ===")


def should_abstain(
    token_probs: list,
    context: str,
    response: str,
    entropy_threshold: float = 1.0,
    faithfulness_threshold: float = 0.5,
) -> dict:
    """
    Decide whether the model should abstain from its response.

    Returns a dict with:
      abstain            : bool
      entropy            : float
      faithfulness       : float
      entropy_flagged    : bool
      faithfulness_flagged: bool
      reason             : str
    """
    entropy = sequence_uncertainty(token_probs)
    faithfulness = faithfulness_score(context, response)

    entropy_flagged      = entropy      > entropy_threshold
    faithfulness_flagged = faithfulness < faithfulness_threshold

    abstain = entropy_flagged or faithfulness_flagged

    reasons = []
    if entropy_flagged:
        reasons.append(f"high uncertainty (entropy={entropy:.2f} > {entropy_threshold})")
    if faithfulness_flagged:
        reasons.append(f"low faithfulness ({faithfulness:.2f} < {faithfulness_threshold})")
    reason = "; ".join(reasons) if reasons else "none"

    return {
        "abstain": abstain,
        "entropy": entropy,
        "faithfulness": faithfulness,
        "entropy_flagged": entropy_flagged,
        "faithfulness_flagged": faithfulness_flagged,
        "reason": reason,
    }


context = "The Eiffel Tower is 330 metres tall and located in Paris, France. It was built in 1889."

scenarios = [
    {
        "label": "Reliable response",
        "token_probs": confident_token_probs,
        "response": "The Eiffel Tower is 330 metres tall and located in Paris.",
    },
    {
        "label": "Uncertain but faithful",
        "token_probs": uncertain_token_probs,
        "response": "The Eiffel Tower is located in Paris and is 330 metres tall.",
    },
    {
        "label": "Confident but hallucinated",
        "token_probs": confident_token_probs,
        "response": "The Eiffel Tower is 500 metres tall and located in Lyon. It won the UNESCO award in 1920.",
    },
    {
        "label": "Uncertain and hallucinated — worst case",
        "token_probs": uncertain_token_probs,
        "response": "The Eiffel Tower is located in Berlin and was built in 1750 by Napoleon.",
    },
]

for s in scenarios:
    result = should_abstain(s["token_probs"], context, s["response"])
    verdict = "ABSTAIN" if result["abstain"] else "RESPOND"
    print(f"\n  [{s['label']}]")
    print(f"    Response  : {s['response'][:70]}")
    print(f"    Entropy   : {result['entropy']:.3f}  Faithfulness: {result['faithfulness']:.2f}")
    print(f"    Decision  : {verdict}")
    if result["reason"] != "none":
        print(f"    Reason    : {result['reason']}")

print("\nKey takeaway:")
print("  Hallucination mitigation = detect uncertainty + verify faithfulness + abstain when unsure.")
print("  Self-consistency is the most powerful single technique for factual QA.")
