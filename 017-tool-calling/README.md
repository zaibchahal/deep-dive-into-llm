# 017 — Tool Calling

## Goal

By the end of this module, you should be able to answer:

* What is tool calling (function calling)?
* How does the LLM decide to call a tool?
* What is the tool call format?
* How is the tool result fed back to the model?
* What are the risks of tool calling?

---

# Theory

## 1. What Is Tool Calling?

LLMs are limited to what's in their training data.

They cannot:

* Search the web in real time
* Execute code
* Look up current weather
* Fetch a database record
* Send an email

Tool calling lets the LLM **request** that an external function be executed.

---

## 2. How It Works

```
User: "What is the weather in Paris today?"

LLM → "I need to call get_weather(city='Paris')"

[System executes get_weather('Paris')]

get_weather → "15°C, sunny"

LLM → "The weather in Paris today is 15°C and sunny."
```

---

## 3. Tool Definition

Tools are described to the LLM in JSON:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "city": {
      "type": "string",
      "description": "City name"
    }
  }
}
```

---

## 4. The Loop

```
1. User sends message
2. LLM sees message + tool definitions
3. LLM decides: answer directly OR call tool
4. If tool: return tool name + arguments
5. System executes tool
6. Tool result sent back to LLM
7. LLM generates final response
```

---

## 5. Structured Output

The LLM produces a **structured** response when calling a tool:

```json
{
  "tool_call": {
    "name": "get_weather",
    "arguments": {
      "city": "Paris"
    }
  }
}
```

This is parsed by the system, not shown to the user.

---

## 6. Multiple Tool Calls

Modern LLMs can call multiple tools in one turn:

```json
[
  {"name": "search", "arguments": {"query": "AI news"}},
  {"name": "get_time", "arguments": {"timezone": "UTC"}}
]
```

---

## 7. Tool Calling in APIs

OpenAI:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=[...],
    tool_choice="auto"
)
```

The model returns `tool_calls` in the response.

---

# Coding Assignments

## Assignment 1 — Define Tools

Create a tool registry:

```python
TOOLS = {
    "get_weather": ...,
    "calculate": ...,
    "search_web": ...,
}
```

---

## Assignment 2 — Implement Tools

```python
def get_weather(city):
    pass

def calculate(expression):
    pass

def get_time(timezone="UTC"):
    pass
```

---

## Assignment 3 — Parse Tool Call

Simulate the LLM outputting a tool call in JSON format.

Parse the JSON and dispatch to the correct function.

---

## Assignment 4 — Tool Call Loop

Implement the full loop:

1. User message
2. LLM decides tool (simulated)
3. Execute tool
4. Feed result back
5. Final response

---

# Success Criteria

* Know the tool calling format (JSON schema)
* Implement a tool registry and dispatcher
* Simulate the full tool-call loop
* Know the risk: LLM can call wrong tools or with wrong args
