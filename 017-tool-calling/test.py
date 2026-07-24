import json
from main import get_weather, calculate, dispatch_tool_call, TOOL_REGISTRY

print("Running tests for 017-tool-calling...")

# Test get_weather returns correct data
result = get_weather("Paris")
assert result["city"] == "Paris"
assert result["temperature"] == 15
assert result["condition"] == "sunny"

# Test get_weather with fahrenheit
result_f = get_weather("Paris", unit="fahrenheit")
assert result_f["temperature"] == 59.0

# Test calculate
result_calc = calculate("2 + 3 * 4")
assert result_calc["result"] == 14

# Test calculate with invalid chars
result_invalid = calculate("import os")
assert "error" in result_invalid

# Test dispatch_tool_call
call_json = json.dumps({"name": "get_weather", "arguments": {"city": "Tokyo"}})
result = dispatch_tool_call(call_json)
assert result["city"] == "Tokyo"

# Test dispatch unknown tool
bad_call = json.dumps({"name": "unknown_tool", "arguments": {}})
result = dispatch_tool_call(bad_call)
assert "error" in result

# Test all tools in registry are callable
for name, fn in TOOL_REGISTRY.items():
    assert callable(fn)

print("All tests passed.")
