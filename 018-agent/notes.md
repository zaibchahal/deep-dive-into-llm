# 018 — Agent: Notes

## Agent vs Chatbot

```
Chatbot: prompt → response (1 turn)
Agent:   observe → think → act → observe → think → act → ...
```

Agents can plan, use tools, and iterate.

## ReAct Pattern

```
Thought: reason about what to do
Action: call a tool
Observation: receive result
→ repeat until done
```

From paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

## Agent Components

- LLM (the brain)
- Tools (hands)
- Memory (short-term: conversation, long-term: persistent)
- Orchestration loop

## Scratchpad

History of thoughts + actions + observations passed back to LLM each step.

The LLM uses this to reason about what it has already done.

## Safety

Always set a max step limit. Without it, an agent can loop forever.

For sensitive actions (send email, delete file), add human approval.

## Memory Types

| Type | Duration | Example |
|------|----------|---------|
| In-context | Single session | Chat history |
| External (DB) | Persistent | User preferences |
| Vector (RAG) | Persistent + searchable | Long-term notes |

## Real Frameworks

- LangChain: popular agent framework
- LlamaIndex: RAG + agents
- AutoGen (Microsoft): multi-agent
- CrewAI: role-based agents
- Cloudflare Agents SDK: stateful agents on Workers

## Next

**SFT** — supervised fine-tuning: train the model on instruction-response pairs.
