# 018 — Agent

## Goal

By the end of this module, you should be able to answer:

* What is an AI agent?
* What is the ReAct loop?
* How does an agent differ from a simple chatbot?
* What are the components of an agent?
* What is an agent scratchpad?

---

# Theory

## 1. What Is an Agent?

A chatbot: one prompt → one response.

An agent: **observe → think → act → observe → think → act → ...**

An agent can:

* Use tools
* Plan multi-step tasks
* Retry on failure
* Decide when it's done

---

## 2. The ReAct Pattern

**ReAct** = Reasoning + Acting (Yao et al., 2022)

```
Thought: I need to find the population of Paris.
Action: search_web("Paris population")
Observation: Paris has a population of 2.16 million.

Thought: Now I know the answer.
Action: finish("Paris has 2.16 million people.")
```

Each cycle:

1. **Thought**: reason about the current state
2. **Action**: call a tool or finish
3. **Observation**: receive the result

---

## 3. Agent vs Chatbot

| | Chatbot | Agent |
|-|---------|-------|
| Turns | 1 | Many |
| Tools | No | Yes |
| Planning | No | Yes |
| Memory | Session | Can persist |
| Autonomy | Low | High |

---

## 4. Agent Loop

```python
while not done:
    thought = llm.think(history)
    
    if thought.is_final_answer:
        done = True
        output = thought.answer
    else:
        tool_result = execute(thought.action)
        history.append(thought)
        history.append(tool_result)
```

---

## 5. The Scratchpad

The agent maintains a scratchpad — the history of thoughts, actions, and observations.

This is passed back to the LLM as context at each step.

---

## 6. System Prompt

The agent's system prompt typically looks like:

```
You are an AI assistant with access to the following tools:
- search_web(query): Search the internet
- calculate(expr): Calculate a math expression
- get_weather(city): Get weather

To use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)

When you have the answer, respond with:
Thought: I now know the answer.
Final Answer: <answer>
```

---

## 7. Safety and Limits

* Max iterations (prevent infinite loops)
* Timeout per action
* Validation of tool arguments
* Human-in-the-loop for sensitive actions

---

# Coding Assignments

## Assignment 1 — Agent Step

```python
def agent_step(query, tools, history):
    thought = think(query, history)
    if thought["type"] == "final":
        return thought["answer"], True
    result = execute_tool(thought["action"], thought["args"])
    return result, False
```

---

## Assignment 2 — Full Agent Loop

```python
def run_agent(query, tools, max_steps=10):
    history = []
    for step in range(max_steps):
        result, done = agent_step(query, tools, history)
        if done:
            return result
    return "Max steps reached"
```

---

## Assignment 3 — Multi-Step Task

Test the agent on a task requiring multiple tool calls:

```
"What is the weather in Paris? And what is 15 * 8?"
```

---

## Assignment 4 — Agent Memory

Give the agent memory of previous conversations.

---

# Success Criteria

* Understand the ReAct loop
* Implement a working agent from scratch
* Handle multi-step tasks
* Implement max step safeguard
