import json
import math
import re

# ---- Tool implementations (same as 017) ----

WEATHER_DB = {
    "paris": {"temp": 15, "condition": "sunny"},
    "london": {"temp": 10, "condition": "cloudy"},
    "tokyo": {"temp": 22, "condition": "partly cloudy"},
    "new york": {"temp": 18, "condition": "rainy"},
}


def get_weather(city):
    data = WEATHER_DB.get(city.lower())
    if not data:
        return f"City '{city}' not found"
    return f"{data['temp']}°C, {data['condition']}"


def calculate(expression):
    allowed = set("0123456789 +-*/().^ ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        expression = expression.replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_web(query):
    simulated = {
        "population": "The world population is approximately 8 billion.",
        "python": "Python is a high-level programming language created in 1991.",
        "gpt": "GPT (Generative Pre-trained Transformer) is a family of LLMs by OpenAI.",
        "transformer": "The Transformer architecture was introduced by Google in 2017.",
        "capital of france": "The capital of France is Paris.",
        "eiffel": "The Eiffel Tower is 330 meters tall, located in Paris.",
    }
    for key, answer in simulated.items():
        if key in query.lower():
            return answer
    return f"Search results for '{query}': Multiple relevant documents found."


TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web,
}


# --------------------------
# Simulated LLM for Agent
# --------------------------

def simulated_llm_think(query, history, tools):
    """
    Simulates the LLM's reasoning step.
    In production: call GPT-4/Claude with the full history as context.
    """
    full_context = query.lower()
    for step in history:
        full_context += " " + str(step).lower()

    answered_weather = any("°c" in str(s).lower() or "condition" in str(s).lower() for s in history)
    answered_calc = any(isinstance(s, dict) and s.get("type") == "observation" and
                        any(c.isdigit() for c in str(s.get("content", "")))
                        for s in history)

    needs_weather = ("weather" in full_context and not answered_weather)
    needs_calc = any(op in query.lower() for op in ["calculate", "compute", "what is", "*", "+", "-", "/", "times"])
    needs_calc = needs_calc and not answered_calc
    needs_search = any(kw in query.lower() for kw in ["who", "what is", "tell me", "explain", "how tall"])
    needs_search = needs_search and "weather" not in query.lower() and "calculate" not in query.lower()

    observations = [s for s in history if isinstance(s, dict) and s.get("type") == "observation"]

    if needs_weather:
        city = "Paris"
        for c in ["london", "tokyo", "new york", "paris"]:
            if c in query.lower():
                city = c.title()
        return {
            "type": "action",
            "thought": f"I need to get the weather for {city}.",
            "tool": "get_weather",
            "args": {"city": city}
        }
    elif needs_calc:
        expr_match = re.search(r'[\d\s\+\-\*\/\.\^]+', query)
        if expr_match:
            expr = expr_match.group().strip()
            return {
                "type": "action",
                "thought": f"I need to calculate: {expr}",
                "tool": "calculate",
                "args": {"expression": expr}
            }
    elif needs_search and not observations:
        return {
            "type": "action",
            "thought": "I need to search for this information.",
            "tool": "search_web",
            "args": {"query": query}
        }

    # Synthesize final answer from observations
    obs_contents = [s["content"] for s in observations]
    answer = "Based on my research: " + " | ".join(obs_contents) if obs_contents else "I can answer directly: " + query

    return {
        "type": "final",
        "thought": "I now have enough information to answer.",
        "answer": answer
    }


# --------------------------
# Assignment 1 — Agent Step
# --------------------------

def agent_step(query, history):
    thought = simulated_llm_think(query, history, TOOLS)

    print(f"  Thought: {thought['thought']}")

    if thought["type"] == "final":
        print(f"  Final Answer: {thought['answer']}")
        return thought["answer"], True

    tool_name = thought["tool"]
    tool_args = thought["args"]
    print(f"  Action: {tool_name}({tool_args})")

    tool_fn = TOOLS.get(tool_name)
    if tool_fn:
        result = tool_fn(**tool_args)
    else:
        result = f"Error: unknown tool {tool_name}"

    print(f"  Observation: {result}")

    history.append({"type": "action", "tool": tool_name, "args": tool_args})
    history.append({"type": "observation", "content": result})

    return result, False


# --------------------------
# Assignment 2 — Full Agent Loop
# --------------------------

def run_agent(query, max_steps=8):
    print(f"\nUser: {query}")
    history = []

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")
        result, done = agent_step(query, history)

        if done:
            print(f"\nFinal: {result}")
            return result

    print("\n[Max steps reached — stopping]")
    return "Max steps reached without final answer."


# --------------------------
# Assignment 3 — Multi-Step Task
# --------------------------

print("=== Assignment 3: Multi-Step Tasks ===")

tasks = [
    "What is the weather in Paris?",
    "Calculate 15 * 8",
    "What is the Eiffel Tower?",
]

for task in tasks:
    run_agent(task, max_steps=5)
    print("\n" + "=" * 60)


# --------------------------
# Assignment 4 — Agent Memory
# --------------------------

print("\n=== Assignment 4: Agent with Memory ===")


class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.long_term_memory = []

    def remember(self, item):
        self.long_term_memory.append(item)

    def recall(self):
        return self.long_term_memory

    def run(self, query, max_steps=5):
        print(f"\n[{self.name}] User: {query}")
        history = list(self.long_term_memory)

        for step in range(max_steps):
            print(f"  [Step {step+1}]")
            result, done = agent_step(query, history)
            if done:
                self.remember({"role": "user", "content": query})
                self.remember({"role": "assistant", "content": result})
                return result

        return "Max steps reached"


agent = Agent("MyAgent")
agent.run("What's the weather in Tokyo?")
agent.run("And what is 100 divided by 4?")

print(f"\nAgent memory has {len(agent.long_term_memory)} items")
