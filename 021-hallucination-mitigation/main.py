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


# --------------------------
# Assignment 5 — Web Search Grounding
# --------------------------

print("\n=== Assignment 5: Web Search Grounding ===")

# Simulated search index: query → list of {url, snippet}
SEARCH_INDEX = {
    "eiffel tower height": [
        {"url": "wikipedia.org/eiffel-tower", "snippet": "The Eiffel Tower is 330 metres tall."},
        {"url": "britannica.com/eiffel-tower", "snippet": "Standing at 330 m, the Eiffel Tower was the world's tallest structure when completed."},
    ],
    "eiffel tower built": [
        {"url": "britannica.com/eiffel-tower", "snippet": "The Eiffel Tower was completed in 1889 for the World's Fair."},
        {"url": "history.com/eiffel-tower",   "snippet": "Construction finished in 1889. It was designed by Gustave Eiffel."},
    ],
    "marie curie nobel": [
        {"url": "nobelprize.org/marie-curie", "snippet": "Marie Curie won two Nobel Prizes: Physics (1903) and Chemistry (1911)."},
        {"url": "wikipedia.org/marie-curie",  "snippet": "She is the only person to have won Nobel Prizes in two different sciences."},
    ],
}

# Ordered from most-specific to least-specific so the right bucket wins.
_INDEX_KEYS = list(SEARCH_INDEX.keys())


def web_search(query: str) -> list:
    """Return simulated search result snippets for the query."""
    key = query.lower().strip()
    for index_key in _INDEX_KEYS:
        keywords = index_key.split()
        if sum(1 for w in keywords if w in key) >= 2:
            return SEARCH_INDEX[index_key]
    return []


def generate_grounded(query: str, search_results: list) -> str:
    """
    Generate a response that only uses information from search snippets.
    (Simulated: concatenates snippet facts into a response sentence.)
    """
    if not search_results:
        return "I don't have reliable information about this query."
    facts = " ".join(r["snippet"] for r in search_results)
    return facts.strip()


def verify_against_search(response: str, search_results: list) -> float:
    """Faithfulness of the response relative to the search snippets."""
    if not search_results:
        return 0.0
    combined_context = " ".join(r["snippet"] for r in search_results)
    return faithfulness_score(combined_context, response)


queries = [
    "How tall is the Eiffel Tower?",
    "When was the Eiffel Tower built?",
    "How many Nobel Prizes did Marie Curie win?",
]

for query in queries:
    results = web_search(query)
    response = generate_grounded(query, results)
    faithfulness = verify_against_search(response, results)
    print(f"\n  Query    : {query}")
    print(f"  Sources  : {len(results)} result(s)")
    if results:
        print(f"  Snippet  : {results[0]['snippet']}")
    print(f"  Response : {response[:80]}")
    print(f"  Faithful : {faithfulness:.2f}")

# Show what happens with a hallucinated (parametric) response vs grounded
print("\n  --- Parametric vs Grounded comparison ---")
eiffel_results = web_search("eiffel tower height")
hallucinated_response = "The Colosseum in Rome is 500 metres tall and was built in 1850 by Napoleon."
grounded_response = generate_grounded("eiffel tower height", eiffel_results)

h_faith = verify_against_search(hallucinated_response, eiffel_results)
g_faith = verify_against_search(grounded_response, eiffel_results)
print(f"  Hallucinated response faithfulness : {h_faith:.2f}")
print(f"  Grounded    response faithfulness  : {g_faith:.2f}")


# --------------------------
# Assignment 6 — Self-Critique
# --------------------------

print("\n=== Assignment 6: Self-Critique ===")

import re

# Each entry: (pattern, label, flags)
# Proper noun check must NOT use IGNORECASE — [A-Z] must mean uppercase only.
HIGH_RISK_PATTERNS = [
    (r'\b\d{4}\b',                                                     "specific year",       re.IGNORECASE),
    (r'\b\d+[\.,]?\d*\s*(metres?|feet|km|miles|kg|lbs|percent|%)\b',  "specific measurement", re.IGNORECASE),
    (r'\b(first|only|largest|smallest|oldest|fastest|tallest|highest|lowest)\b', "superlative claim", re.IGNORECASE),
    (r'(?<!\. )\b[A-Z][a-z]{2,} [A-Z][a-z]{2,}\b',                   "proper noun",         0),
    (r'\bin \d{4}\b',                                                  "dated event",         re.IGNORECASE),
]


def self_critique(response: str) -> dict:
    """
    Scan a response for high-risk tokens and flag uncertain claims.

    Returns:
      flagged_claims : list of (sentence, [risk_types]) for risky sentences
      risk_score     : float 0-1 (fraction of sentences flagged)
      critique       : human-readable summary
    """
    sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if not sentences:
        return {"flagged_claims": [], "risk_score": 0.0, "critique": "No claims to critique."}

    flagged = []
    for sentence in sentences:
        risks = []
        for pattern, label, flags in HIGH_RISK_PATTERNS:
            if re.search(pattern, sentence, flags):
                risks.append(label)
        if risks:
            flagged.append((sentence, list(dict.fromkeys(risks))))  # deduplicate

    risk_score = len(flagged) / len(sentences)

    if risk_score == 0:
        critique = "No high-risk claims detected. Response appears safe."
    elif risk_score < 0.5:
        critique = f"{len(flagged)}/{len(sentences)} sentences contain specific claims that should be verified."
    else:
        critique = f"High risk: {len(flagged)}/{len(sentences)} sentences contain specific claims. Consider re-checking with a source."

    return {"flagged_claims": flagged, "risk_score": risk_score, "critique": critique}


responses_to_critique = [
    "The Eiffel Tower was built in 1889 by Gustave Eiffel and stands 330 metres tall. It is the tallest structure in Paris.",
    "Marie Curie won two Nobel Prizes, in 1903 and 1911, making her the only person to win in two different sciences.",
    "Exercise is generally beneficial for health and wellbeing.",
    "The telephone was invented by Thomas Edison in 1876 and quickly became the most popular communication device.",
]

for resp in responses_to_critique:
    result = self_critique(resp)
    print(f"\n  Response  : {resp[:80]}")
    print(f"  Risk score: {result['risk_score']:.2f}")
    print(f"  Critique  : {result['critique']}")
    for claim, risks in result["flagged_claims"]:
        print(f"    ⚑ '{claim[:60]}' → {', '.join(risks)}")


# --------------------------
# Assignment 7 — Citation Enforcement
# --------------------------

print("\n=== Assignment 7: Citation Enforcement ===")


def check_citations(response: str, sources: list) -> dict:
    """
    Verify citation coverage and faithfulness for each cited source.

    sources: list of {id, url, snippet}

    Returns:
      uncited_sentences     : factual sentences with no [N] citation marker
      unsupported_citations : (sentence, source) pairs where snippet doesn't support the claim
      citation_coverage     : fraction of sentences that carry a citation
    """
    source_map = {str(s["id"]): s for s in sources}

    sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    uncited = []
    unsupported = []
    cited_count = 0

    citation_re = re.compile(r'\[(\d+)\]')

    for sentence in sentences:
        refs = citation_re.findall(sentence)
        clean = citation_re.sub('', sentence).strip()

        if not refs:
            uncited.append(sentence)
        else:
            cited_count += 1
            for ref in refs:
                source = source_map.get(ref)
                if source is None:
                    unsupported.append((sentence, f"[{ref}] does not exist"))
                else:
                    # Check if the source snippet actually supports the claim
                    support = faithfulness_score(source["snippet"], clean)
                    if support < 0.4:
                        unsupported.append((sentence, source["url"]))

    coverage = cited_count / len(sentences) if sentences else 0.0

    return {
        "uncited_sentences": uncited,
        "unsupported_citations": unsupported,
        "citation_coverage": coverage,
    }


sources = [
    {"id": 1, "url": "wikipedia.org/eiffel-tower",  "snippet": "The Eiffel Tower is 330 metres tall and located in Paris, France."},
    {"id": 2, "url": "britannica.com/eiffel-tower",  "snippet": "The Eiffel Tower was completed in 1889 for the World's Fair."},
    {"id": 3, "url": "history.com/eiffel-tower",     "snippet": "It was designed by the engineer Gustave Eiffel."},
]

citation_examples = [
    {
        "label": "Well-cited, faithful",
        "response": "The Eiffel Tower is 330 metres tall [1]. It was completed in 1889 [2]. It was designed by Gustave Eiffel [3].",
    },
    {
        "label": "Missing citation on last claim",
        "response": "The Eiffel Tower is 330 metres tall [1]. It was completed in 1889 [2]. It is the most visited monument in the world.",
    },
    {
        "label": "Citation present but claim not supported by source",
        "response": "The Eiffel Tower is 500 metres tall [1]. It was built in 1750 [2].",
    },
    {
        "label": "No citations at all",
        "response": "The Eiffel Tower is 330 metres tall. It was completed in 1889. Gustave Eiffel designed it.",
    },
]

for ex in citation_examples:
    result = check_citations(ex["response"], sources)
    print(f"\n  [{ex['label']}]")
    print(f"    Citation coverage : {result['citation_coverage']:.0%}")
    if result["uncited_sentences"]:
        print(f"    Uncited           : {len(result['uncited_sentences'])} sentence(s)")
        for s in result["uncited_sentences"]:
            print(f"      - '{s[:60]}'")
    if result["unsupported_citations"]:
        print(f"    Unsupported cites : {len(result['unsupported_citations'])}")
        for sentence, source in result["unsupported_citations"]:
            print(f"      - '{sentence[:50]}' → {source}")

print("\nKey takeaway:")
print("  Web search grounds claims in live facts.")
print("  Self-critique flags risky claims without any external call.")
print("  Citation enforcement catches both missing and misattributed sources.")
