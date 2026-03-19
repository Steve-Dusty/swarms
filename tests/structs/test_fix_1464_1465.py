"""
Tests for fix of issues #1464 and #1465:
- #1464: _run() no longer silently swallows exceptions
- #1465: Errors are logged exactly once, not double-logged
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange


def create_agents():
    """Create real agents for testing."""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics",
            system_prompt="You are a research specialist. Provide concise answers.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
        ),
        Agent(
            agent_name="WriterAgent",
            agent_description="Expert in writing content",
            system_prompt="You are a writing expert. Provide concise answers.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
        ),
    ]


def make_rearrange(agents, flow, **kwargs):
    return AgentRearrange(agents=agents, flow=flow, max_loops=1, **kwargs)


# ============================================================
# Issue #1464: Exceptions must propagate, not return None
# ============================================================


def test_missing_agent_raises():
    """run() must raise when flow references a removed agent."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r.run("test")


def test_broken_conversation_raises():
    """run() must raise when conversation is corrupted."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")
    r.conversation.conversation_history = None

    with pytest.raises((TypeError, AttributeError)):
        r.run("test")


def test_agent_error_raises():
    """run() must raise when an agent's run() raises unexpectedly."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")

    original_run = r.agents["WriterAgent"].run

    def bad_run(*args, **kwargs):
        raise TypeError("unexpected error in agent")

    r.agents["WriterAgent"].run = bad_run

    with pytest.raises(TypeError, match="unexpected error in agent"):
        r.run("test")


def test_callable_propagates():
    """__call__ must raise, not return the exception object."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r("test")


def test_batch_run_propagates():
    """batch_run must raise, not return None."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r.batch_run(["test1", "test2"])


# ============================================================
# Issue #1465: Errors logged exactly once
# ============================================================


def test_error_logged_once():
    """_catch_error should fire exactly once per failure."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

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
# Existing behavior: successful runs still work with real LLMs
# ============================================================


def test_successful_run_returns_result():
    """A successful run with real LLM must return a non-None result."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")

    result = r.run("What is 2+2?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_successful_callable_returns_result():
    """__call__ with real LLM on success must return a result."""
    agents = create_agents()
    r = make_rearrange(agents, "ResearchAgent -> WriterAgent")

    result = r("What is 2+2?")
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
