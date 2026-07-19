# agent-tasks

A minimal tool-using agent runtime with an emphasis on debuggability.

## Problem

Most agent frameworks are black boxes: the model calls some tools, an answer
comes out, and when the result is wrong you have very little to work with.
Which tool was called? With what arguments? What did it return? How many
steps did the loop burn before giving up?

`agent-tasks` is my attempt at the opposite: a small runtime where **every
step — decision, tool call, observation — is traced to SQLite** as it
happens, so a run can be reconstructed and debugged after the fact.

## Design

The loop is deliberately boring:

```
task -> [plan (LLM decides next step) -> dispatch tool -> observe] * n -> final answer
```

- `agent/runtime.py` — the agent loop. A `Tool` registry (JSON schemas via
  pydantic), dispatch with argument validation, retry with exponential
  backoff on tool errors, and a hard `max_steps` guard so a confused model
  can't loop forever.
- `agent/trace.py` — a `Tracer` that writes runs and steps to SQLite. No
  framework, just `sqlite3`.
- `agent/tools.py` — a few real local tools (calculator, word count, read
  file) so the loop has something to chew on.
- The LLM sits behind a small `LLMProvider` interface
  (`decide(messages) -> dict`). `MockProvider` is deterministic and scripted,
  so the whole loop — including tracing — runs in tests with **no API key**.
  `OpenAIProvider` is the thin real implementation behind the same interface.

## Status

**Early — building in public.** The loop, tracing, mock provider, and tests
genuinely work (see below). The OpenAI provider is written but not yet
exercised end-to-end. Expect sharp edges.

## Roadmap

- [x] Agent loop: plan → tool call → observe
- [x] Tool registry with pydantic JSON schemas
- [x] Retry with backoff on tool errors
- [x] `max_steps` guard
- [x] SQLite tracing of every step
- [x] Deterministic mock provider + pytest coverage
- [ ] Exercise `OpenAIProvider` end-to-end against the real API
- [ ] CLI entry point (`python -m agent "task"`)
- [ ] Trace inspection tooling (dump a run as a readable timeline)
- [ ] Sandboxed `read_file` (restrict to a working directory)
- [ ] Async tool execution

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests (no API key needed — they use the mock provider):

```bash
pytest
```

To run against the real OpenAI API, copy `.env.example` to `.env` and fill
in `OPENAI_API_KEY`.
