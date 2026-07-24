import json
import math
from datetime import datetime

# --------------------------
# Assignment 1 — Define Tools
# --------------------------

print("=== Assignment 1: Tool Definitions ===")

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "city": {"type": "string", "description": "Name of the city"},
            "unit": {"type": "string", "description": "Temperature unit: celsius or fahrenheit", "default": "celsius"}
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "expression": {"type": "string", "description": "A math expression like '2 + 3 * 4'"}
        }
    },
    {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {
            "timezone": {"type": "string", "description": "Timezone name", "default": "UTC"}
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    }
]

for tool in TOOL_DEFINITIONS:
    print(f"  Tool: {tool['name']} — {tool['description']}")


# --------------------------
# Assignment 2 — Implement Tools
# --------------------------

print("\n=== Assignment 2: Tool Implementations ===")

WEATHER_DB = {
    "paris": {"temp": 15, "condition": "sunny", "humidity": 60},
    "london": {"temp": 10, "condition": "cloudy", "humidity": 75},
    "tokyo": {"temp": 22, "condition": "partly cloudy", "humidity": 65},
    "new york": {"temp": 18, "condition": "rainy", "humidity": 80},
    "sydney": {"temp": 25, "condition": "clear", "humidity": 55},
}


def get_weather(city, unit="celsius"):
    data = WEATHER_DB.get(city.lower())
    if not data:
        return {"error": f"City '{city}' not found"}
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = temp * 9/5 + 32
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"]
    }


def calculate(expression):
    allowed = set("0123456789 +-*/().,^ ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        expression = expression.replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def get_time(timezone="UTC"):
    now = datetime.utcnow()
    return {
        "timezone": timezone,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "note": "This is UTC time (timezone offset not applied in simulation)"
    }


def search_web(query):
    simulated_results = {
        "python": "Python is a programming language created by Guido van Rossum.",
        "ai": "Artificial intelligence is intelligence demonstrated by machines.",
        "transformer": "Transformer is a neural network architecture using self-attention.",
    }
    for key, result in simulated_results.items():
        if key in query.lower():
            return {"query": query, "result": result}
    return {"query": query, "result": f"Found 1,240 results for '{query}'"}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_time": get_time,
    "search_web": search_web,
}

print("Testing tools:")
print("get_weather('Paris'):", get_weather("Paris"))
print("calculate('2 + 3 * 4'):", calculate("2 + 3 * 4"))
print("get_time():", get_time())
print("search_web('python'):", search_web("python"))


# --------------------------
# Assignment 3 — Parse Tool Call
# --------------------------

print("\n=== Assignment 3: Parse and Dispatch Tool Call ===")

def dispatch_tool_call(tool_call_json):
    """Parse a JSON tool call string and execute the corresponding function."""
    tool_call = json.loads(tool_call_json)
    name = tool_call["name"]
    arguments = tool_call.get("arguments", {})

    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}

    return TOOL_REGISTRY[name](**arguments)


simulated_tool_calls = [
    '{"name": "get_weather", "arguments": {"city": "London"}}',
    '{"name": "calculate", "arguments": {"expression": "10 * 20 + 5"}}',
    '{"name": "get_time", "arguments": {}}',
    '{"name": "search_web", "arguments": {"query": "transformer neural network"}}',
]

for call in simulated_tool_calls:
    result = dispatch_tool_call(call)
    print(f"Call: {call[:50]}...")
    print(f"Result: {result}\n")


# --------------------------
# Assignment 4 — Tool Call Loop
# --------------------------

print("=== Assignment 4: Full Tool Call Loop ===")

def simulated_llm_with_tools(message, tool_definitions):
    """
    Simulates an LLM that can decide to call tools.
    In production: call OpenAI/Anthropic API with tool definitions.
    """
    message_lower = message.lower()

    if "weather" in message_lower:
        city = "Paris"
        for c in ["london", "tokyo", "new york", "sydney", "paris"]:
            if c in message_lower:
                city = c.title()
                break
        return {
            "type": "tool_call",
            "tool_call": {"name": "get_weather", "arguments": {"city": city}}
        }
    elif any(op in message_lower for op in ["calculate", "compute", "what is", "math"]):
        import re
        expr = re.search(r'[\d\s\+\-\*\/\.\^\(\)]+', message)
        if expr:
            return {
                "type": "tool_call",
                "tool_call": {"name": "calculate", "arguments": {"expression": expr.group().strip()}}
            }
    elif "time" in message_lower or "date" in message_lower:
        return {
            "type": "tool_call",
            "tool_call": {"name": "get_time", "arguments": {}}
        }

    return {
        "type": "text",
        "content": f"I'll answer directly: {message}"
    }


def simulated_llm_final_response(message, tool_result):
    """Simulate LLM generating final answer after seeing tool result."""
    name = list(tool_result.keys())[0] if tool_result else "result"
    return f"Based on the tool result: {json.dumps(tool_result)}"


def tool_call_loop(user_message):
    print(f"User: {user_message}")

    llm_response = simulated_llm_with_tools(user_message, TOOL_DEFINITIONS)

    if llm_response["type"] == "text":
        print(f"Assistant: {llm_response['content']}")
        return

    tool_call = llm_response["tool_call"]
    print(f"[LLM calls tool: {tool_call['name']}({tool_call['arguments']})]")

    tool_fn = TOOL_REGISTRY.get(tool_call["name"])
    tool_result = tool_fn(**tool_call["arguments"])
    print(f"[Tool result: {tool_result}]")

    final_answer = simulated_llm_final_response(user_message, tool_result)
    print(f"Assistant: {final_answer}")


messages = [
    "What's the weather in London?",
    "Calculate 15 * 8 + 32",
    "What time is it?",
    "Tell me about Python.",
]

for msg in messages:
    print("\n" + "─" * 50)
    tool_call_loop(msg)
