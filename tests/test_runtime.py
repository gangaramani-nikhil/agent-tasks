"""Tests for the agent loop, run end-to-end against the deterministic
MockProvider with tracing into a temporary SQLite database."""
import sqlite3

import pytest
from pydantic import BaseModel

from agent.runtime import Agent, MaxStepsExceeded, MockProvider, Tool
from agent.tools import calculator, default_tools, read_file, word_count
from agent.trace import Tracer


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces.db")


def make_agent(tracer, script, **kwargs):
    provider = MockProvider(script)
    return Agent(provider, default_tools(), tracer, **kwargs), provider


def test_loop_calls_tool_then_finishes(tracer):
    agent, provider = make_agent(
        tracer,
        [
            {"tool": "calculator", "args": {"expression": "2 * (3 + 4)"}},
            {"final": "2 * (3 + 4) = 14"},
        ],
    )
    answer = agent.run("what is 2 * (3 + 4)?")

    assert answer == "2 * (3 + 4) = 14"
    # The second decide() call must have seen the tool's observation.
    assert len(provider.calls) == 2
    tool_messages = [m for m in provider.calls[1] if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "14"


def test_every_step_is_traced(tracer):
    agent, _ = make_agent(
        tracer,
        [
            {"tool": "word_count", "args": {"text": "hello world"}},
            {"final": "2 words"},
        ],
    )
    agent.run("count the words")

    with sqlite3.connect(tracer.db_path) as conn:
        run = conn.execute("SELECT task, final, ended_at FROM runs").fetchone()
        steps = conn.execute(
            "SELECT step, kind FROM steps ORDER BY id"
        ).fetchall()

    assert run[0] == "count the words"
    assert run[1] == "2 words"
    assert run[2] is not None  # ended_at was recorded
    # step 0: decision + observation, step 1: final decision
    assert steps == [(0, "decision"), (0, "observation"), (1, "decision")]


def test_max_steps_guard(tracer):
    # A provider that keeps calling tools forever must be stopped.
    agent, _ = make_agent(
        tracer,
        [{"tool": "calculator", "args": {"expression": "1 + 1"}}],
        max_steps=3,
    )
    agent.provider.decide = lambda messages: {
        "tool": "calculator",
        "args": {"expression": "1 + 1"},
    }

    with pytest.raises(MaxStepsExceeded):
        agent.run("loop forever")


def test_tool_error_is_retried_and_recovered(tracer):
    attempts = {"n": 0}

    class FlakyArgs(BaseModel):
        pass

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient failure")
        return "recovered"

    flaky_tool = Tool(
        name="flaky", description="fails once, then works", args_model=FlakyArgs, fn=flaky
    )
    agent = Agent(
        MockProvider(
            [{"tool": "flaky", "args": {}}, {"final": "ok"}],
        ),
        [flaky_tool],
        tracer,
        retries=2,
        backoff=0,  # no sleeping in tests
    )

    assert agent.run("try the flaky tool") == "ok"
    assert attempts["n"] == 2  # one failure + one successful retry


def test_exhausted_retries_become_an_error_observation(tracer):
    class FlakyArgs(BaseModel):
        pass

    def always_fails() -> str:
        raise RuntimeError("permanent failure")

    tool = Tool(
        name="broken", description="always fails", args_model=FlakyArgs, fn=always_fails
    )
    provider = MockProvider([{"tool": "broken", "args": {}}, {"final": "gave up"}])
    agent = Agent(provider, [tool], tracer, retries=1, backoff=0)

    assert agent.run("call the broken tool") == "gave up"
    tool_messages = [m for m in provider.calls[1] if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "error: permanent failure"


def test_unknown_tool_returns_error_observation(tracer):
    agent, provider = make_agent(
        tracer,
        [{"tool": "does_not_exist", "args": {}}, {"final": "handled"}],
    )
    assert agent.run("call a missing tool") == "handled"
    tool_messages = [m for m in provider.calls[1] if m["role"] == "tool"]
    assert "unknown tool" in tool_messages[0]["content"]


def test_invalid_args_return_validation_error(tracer):
    agent, provider = make_agent(
        tracer,
        [
            {"tool": "calculator", "args": {"wrong_field": 1}},  # missing 'expression'
            {"final": "handled"},
        ],
        retries=0,
    )
    assert agent.run("bad args") == "handled"
    tool_messages = [m for m in provider.calls[1] if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("error:")


# --- the built-in tools themselves ------------------------------------------


def test_calculator_arithmetic():
    assert calculator("2 * (3 + 4)") == "14"
    assert calculator("10 / 4") == "2.5"
    assert calculator("2 ** 10") == "1024"


def test_calculator_rejects_unsafe_expressions():
    with pytest.raises(Exception):
        calculator("__import__('os').system('echo hi')")


def test_word_count():
    assert word_count("hello world") == {"words": 2, "lines": 1, "characters": 11}
    assert word_count("a\nb\nc")["lines"] == 3


def test_read_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello from a file")
    assert read_file(str(f)) == "hello from a file"
    with pytest.raises(FileNotFoundError):
        read_file(str(tmp_path / "missing.txt"))
