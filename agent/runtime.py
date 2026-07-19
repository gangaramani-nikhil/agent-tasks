"""The agent loop: plan -> tool call -> observe, with every step traced.

The LLM sits behind the `LLMProvider` interface: given the conversation so
far it returns either {"tool": <name>, "args": {...}} or {"final": <answer>}.
`MockProvider` is a scripted, deterministic implementation so the whole loop
runs in tests without an API key; `OpenAIProvider` is the real one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from .trace import Tracer


class MaxStepsExceeded(RuntimeError):
    """Raised when the agent burns through max_steps without a final answer."""


class LLMProvider(Protocol):
    """Anything that can decide the agent's next step."""

    def decide(self, messages: list[dict]) -> dict:
        """Return {"tool": name, "args": {...}} or {"final": answer}."""
        ...


@dataclass
class Tool:
    """A callable tool: name, description, JSON schema, and implementation."""

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }

    def run(self, args: dict) -> Any:
        """Validate raw args against the schema, then call the function."""
        parsed = self.args_model.model_validate(args)
        return self.fn(**parsed.model_dump())


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool],
        tracer: Tracer,
        max_steps: int = 10,
        retries: int = 2,
        backoff: float = 0.1,
    ) -> None:
        self.provider = provider
        self.registry = {t.name: t for t in tools}
        self.tracer = tracer
        self.max_steps = max_steps
        self.retries = retries
        self.backoff = backoff

    def _call_with_retry(self, tool: Tool, args: dict) -> str:
        """Run a tool, retrying with exponential backoff on errors.

        After retries are exhausted the error becomes the observation, so the
        agent sees what went wrong instead of crashing the whole run.
        """
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return str(tool.run(args))
            except Exception as exc:  # validation errors included
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        return f"error: {last_error}"

    def run(self, task: str) -> str:
        run_id = self.tracer.start_run(task)
        messages: list[dict] = [{"role": "user", "content": task}]

        for step in range(self.max_steps):
            decision = self.provider.decide(messages)
            self.tracer.log(run_id, step, "decision", decision)

            if "final" in decision:
                self.tracer.end_run(run_id, str(decision["final"]))
                return str(decision["final"])

            name = decision.get("tool", "")
            args = decision.get("args", {})
            tool = self.registry.get(name)
            if tool is None:
                observation = f"error: unknown tool '{name}'"
            else:
                observation = self._call_with_retry(tool, args)

            self.tracer.log(
                run_id, step, "observation", {"tool": name, "result": observation}
            )
            messages.append({"role": "assistant", "content": json.dumps(decision)})
            messages.append({"role": "tool", "name": name, "content": observation})

        raise MaxStepsExceeded(
            f"agent did not finish within {self.max_steps} steps (run_id={run_id})"
        )


class MockProvider:
    """Deterministic, scripted provider for tests and local development.

    Pops one scripted decision per call. When the script runs out it answers
    with `fallback_final`, so a script only needs to cover the tool calls.
    """

    def __init__(self, script: list[dict], fallback_final: str = "done (mock)") -> None:
        self._script = list(script)
        self._fallback_final = fallback_final
        self.calls: list[list[dict]] = []  # messages seen on each decide()

    def decide(self, messages: list[dict]) -> dict:
        self.calls.append(list(messages))
        if self._script:
            return self._script.pop(0)
        return {"final": self._fallback_final}


class OpenAIProvider:
    """Real provider backed by the OpenAI chat-completions tool-calling API.

    TODO: not yet exercised end-to-end against the live API.
    """

    def __init__(self, tools: list[Tool], model: str | None = None) -> None:
        from openai import OpenAI  # imported lazily so tests don't need the dep configured

        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_model.model_json_schema(),
                },
            }
            for t in tools
        ]

    def decide(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            return {
                "tool": call.function.name,
                "args": json.loads(call.function.arguments or "{}"),
            }
        return {"final": message.content or ""}
