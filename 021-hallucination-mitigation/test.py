import numpy as np
from main import (
    majority_vote, token_entropy, sequence_uncertainty,
    faithfulness_score, should_abstain,
    web_search, generate_grounded, verify_against_search,
    self_critique, check_citations,
)

print("Running tests for 021-hallucination-mitigation...")

# --- majority_vote ---

winner, agreement = majority_vote(["A", "A", "B", "A", "C"])
assert winner == "A", f"Expected 'A', got '{winner}'"
assert abs(agreement - 0.6) < 1e-9, f"Expected 0.6, got {agreement}"

winner, agreement = majority_vote(["yes", "YES", "Yes", "no"])
assert winner.strip().lower() == "yes"
assert abs(agreement - 0.75) < 1e-9

# Single sample
winner, agreement = majority_vote(["only answer"])
assert agreement == 1.0

# --- token_entropy ---

# Uniform over 4 tokens → entropy = log2(4) = 2 bits
uniform = np.array([0.25, 0.25, 0.25, 0.25])
assert abs(token_entropy(uniform) - 2.0) < 1e-6, f"Uniform entropy should be 2.0, got {token_entropy(uniform)}"

# Deterministic → entropy = 0
certain = np.array([1.0, 0.0, 0.0, 0.0])
assert token_entropy(certain) < 1e-6, f"Certain entropy should be ~0, got {token_entropy(certain)}"

# Entropy is non-negative
mixed = np.array([0.7, 0.2, 0.1])
assert token_entropy(mixed) >= 0

# --- sequence_uncertainty ---

# All deterministic tokens → near 0
all_certain = [np.array([1.0, 0.0, 0.0]) for _ in range(5)]
assert sequence_uncertainty(all_certain) < 1e-6

# All uniform (4 choices) → 2 bits
all_uniform = [np.array([0.25, 0.25, 0.25, 0.25]) for _ in range(5)]
assert abs(sequence_uncertainty(all_uniform) - 2.0) < 1e-6

# Uncertain > confident
confident = [np.array([0.9, 0.05, 0.05]) for _ in range(3)]
uncertain = [np.array([0.35, 0.35, 0.30]) for _ in range(3)]
assert sequence_uncertainty(uncertain) > sequence_uncertainty(confident)

# --- faithfulness_score ---

ctx = "The Eiffel Tower is 330 metres tall and located in Paris France and was built in 1889."

# Fully supported response → high score
resp_good = "The Eiffel Tower is 330 metres tall. It was built in 1889."
score_good = faithfulness_score(ctx, resp_good)
assert score_good >= 0.5, f"Good response faithfulness should be >= 0.5, got {score_good}"

# Fully fabricated response → low score
resp_bad = "The Eiffel Tower is located in Berlin. It was designed by Leonardo da Vinci in 1750."
score_bad = faithfulness_score(ctx, resp_bad)
assert score_bad <= score_good, f"Bad response should score <= good response"

# Score is in [0, 1]
assert 0.0 <= score_good <= 1.0
assert 0.0 <= score_bad  <= 1.0

# --- should_abstain ---

ctx2 = "The capital of France is Paris."
certain_probs = [np.array([0.95, 0.03, 0.02]) for _ in range(3)]
uncertain_probs = [np.array([0.34, 0.33, 0.33]) for _ in range(3)]

# Certain + faithful → should NOT abstain
result = should_abstain(certain_probs, ctx2, "The capital of France is Paris.", entropy_threshold=1.0, faithfulness_threshold=0.5)
assert not result["abstain"], "Should not abstain on certain + faithful response"

# Uncertain + faithful → should abstain (entropy)
result = should_abstain(uncertain_probs, ctx2, "The capital of France is Paris.", entropy_threshold=1.0, faithfulness_threshold=0.5)
assert result["abstain"], "Should abstain on uncertain response"
assert result["entropy_flagged"]

# Certain + unfaithful → should abstain (faithfulness)
result = should_abstain(certain_probs, ctx2, "The capital of Germany is Berlin and it is very large.", entropy_threshold=1.0, faithfulness_threshold=0.5)
assert result["abstain"], "Should abstain on unfaithful response"
assert result["faithfulness_flagged"]

# --- web_search ---

results = web_search("how tall is the eiffel tower height")
assert isinstance(results, list), "web_search should return a list"
assert len(results) > 0, "should find results for eiffel tower height"
assert "url" in results[0] and "snippet" in results[0]

# Unknown query returns empty list
assert web_search("xyzzy nonexistent query 999") == []

# --- generate_grounded ---

results = web_search("how tall is the eiffel tower height")
response = generate_grounded("how tall is the Eiffel Tower", results)
assert isinstance(response, str) and len(response) > 0

# No results → abstention message
empty_response = generate_grounded("unknown query", [])
assert "don't" in empty_response.lower() or "reliable" in empty_response.lower()

# --- verify_against_search ---

results = web_search("how tall is the eiffel tower height")
good_resp = "The Eiffel Tower is 330 metres tall."
bad_resp  = "The Colosseum in Rome was built by Napoleon in 1850."
assert verify_against_search(good_resp, results) >= verify_against_search(bad_resp, results)

# No results → 0.0
assert verify_against_search("anything", []) == 0.0

# --- self_critique ---

risky = "Marie Curie won two Nobel Prizes in 1903 and 1911."
safe  = "Exercise is generally considered healthy."

risky_result = self_critique(risky)
safe_result  = self_critique(safe)

assert risky_result["risk_score"] > safe_result["risk_score"], \
    f"Risky response should score higher than safe. Got {risky_result['risk_score']} vs {safe_result['risk_score']}"
assert 0.0 <= risky_result["risk_score"] <= 1.0
assert 0.0 <= safe_result["risk_score"]  <= 1.0
assert isinstance(risky_result["flagged_claims"], list)
assert isinstance(risky_result["critique"], str)

# Empty response
empty_result = self_critique("")
assert empty_result["risk_score"] == 0.0

# --- check_citations ---

sources = [
    {"id": 1, "url": "wiki.org/eiffel", "snippet": "The Eiffel Tower is 330 metres tall and located in Paris."},
    {"id": 2, "url": "brit.com/eiffel", "snippet": "The Eiffel Tower was completed in 1889."},
]

# Fully cited, faithful → high coverage, no uncited, no unsupported
result = check_citations(
    "The Eiffel Tower is 330 metres tall [1]. It was completed in 1889 [2].",
    sources,
)
assert result["citation_coverage"] == 1.0, f"Expected 1.0, got {result['citation_coverage']}"
assert len(result["uncited_sentences"]) == 0

# One uncited sentence → coverage < 1
result2 = check_citations(
    "The Eiffel Tower is 330 metres tall [1]. It is the most visited monument in the world.",
    sources,
)
assert result2["citation_coverage"] < 1.0
assert len(result2["uncited_sentences"]) == 1

# No citations at all → coverage 0
result3 = check_citations(
    "The Eiffel Tower is 330 metres tall. It was completed in 1889.",
    sources,
)
assert result3["citation_coverage"] == 0.0
assert len(result3["uncited_sentences"]) == 2

print("All tests passed.")
