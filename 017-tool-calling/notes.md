# 017 — Tool Calling: Notes

## Why Tool Calling?

LLMs have no access to:
- Real-time data
- Private databases
- Code execution
- External APIs

Tool calling bridges the LLM's text world with real actions.

## The Loop

```
User message
  → LLM (with tool definitions)
  → tool_call: {name, arguments}
  → Execute tool
  → Tool result
  → LLM (with tool result)
  → Final answer
```

## Tool Definition Format

```json
{
  "name": "get_weather",
  "description": "...",
  "parameters": {
    "city": {"type": "string", "description": "..."}
  }
}
```

The description is what the LLM reads to decide when to call the tool.

## Structured Output

The LLM returns JSON (not natural language) for the tool call.

```json
{"name": "get_weather", "arguments": {"city": "Paris"}}
```

Your code parses this and dispatches to the real function.

## Risks

- LLM can call wrong tool
- LLM can pass wrong arguments
- Tool can fail — need error handling
- Security: never eval untrusted code without sanitization

## Real Implementation

```python
# OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tool_definitions,
    tool_choice="auto"
)
if response.choices[0].message.tool_calls:
    # Execute the tool
    ...
```

## Next

**Agent** — chain multiple tool calls in a reasoning loop (observe → think → act).
