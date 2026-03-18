"""
Tests for fix of issues #1464 and #1465:
- #1464: _run() no longer silently swallows exceptions
- #1465: Errors are logged exactly once, not double-logged
"""

import pytest
from unittest.mock import patch
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange


class FakeLLM:
    def run(self, task: str, *args, **kwargs) -> str:
        return "fake response"


def make_rearrange(agents, flow, **kwargs):
    return AgentRearrange(agents=agents, flow=flow, max_loops=1, **kwargs)


# ============================================================
# Issue #1464: Exceptions must propagate, not return None
# ============================================================


def test_missing_agent_raises():
    """run() must raise when flow references a removed agent."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    b = Agent(agent_name="B", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a, b], "A -> B")
    del r.agents["B"]

    with pytest.raises(ValueError, match="not registered"):
        r.run("test")


def test_broken_conversation_raises():
    """run() must raise when conversation is corrupted."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a], "A -> A")
    r.conversation.conversation_history = None  # sabotage

    with pytest.raises((TypeError, AttributeError)):
        r.run("test")


def test_agent_error_raises():
    """run() must raise when an agent's run() raises unexpectedly."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a], "A -> A")

    def bad_run(*args, **kwargs):
        raise TypeError("unexpected error")
    r.agents["A"].run = bad_run

    with pytest.raises(TypeError, match="unexpected error"):
        r.run("test")


def test_callable_propagates():
    """__call__ must raise, not return the exception object."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    b = Agent(agent_name="B", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a, b], "A -> B")
    del r.agents["B"]

    with pytest.raises(ValueError, match="not registered"):
        r("test")


def test_batch_run_propagates():
    """batch_run must raise, not return None."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    b = Agent(agent_name="B", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a, b], "A -> B")
    del r.agents["B"]

    with pytest.raises(ValueError, match="not registered"):
        r.batch_run(["test1", "test2"])


# ============================================================
# Issue #1465: Errors logged exactly once
# ============================================================


def test_error_logged_once():
    """_catch_error should fire exactly once per failure."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    b = Agent(agent_name="B", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a, b], "A -> B")
    del r.agents["B"]

    call_count = 0
    original_catch = r._catch_error

    def counting_catch(e):
        nonlocal call_count
        call_count += 1
        original_catch(e)

    r._catch_error = counting_catch

    with pytest.raises(ValueError):
        r.run("test")

    assert call_count == 1, f"_catch_error called {call_count} times, expected 1"


# ============================================================
# Existing behavior: successful runs still work
# ============================================================


def test_successful_run_returns_result():
    """A successful run must still return a non-None result."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a], "A -> A")

    result = r.run("test task")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_successful_callable_returns_result():
    """__call__ on success must return a result."""
    a = Agent(agent_name="A", llm=FakeLLM(), max_loops=1)
    r = make_rearrange([a], "A -> A")

    result = r("test task")
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
