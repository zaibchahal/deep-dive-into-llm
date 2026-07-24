from main import run_agent, Agent, TOOLS, get_weather, calculate

print("Running tests for 018-agent...")

# Test individual tools work
assert "°C" in get_weather("Paris") or "°c" in get_weather("Paris").lower()
assert calculate("2 + 2") == "4"

# Test agent runs without error
result = run_agent("What is the weather in London?", max_steps=5)
assert result is not None
assert len(result) > 0

# Test agent memory
agent = Agent("TestAgent")
agent.run("What's the weather in Tokyo?", max_steps=5)
assert len(agent.long_term_memory) >= 2, "Agent should store conversation in memory"

# Test max steps safeguard
result_ms = run_agent("What is the meaning of life?", max_steps=2)
assert result_ms is not None

print("All tests passed.")
