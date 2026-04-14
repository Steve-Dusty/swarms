"""Tests for AdvisorSwarm.

Uses real agents and API calls — no mocks.
"""

import pytest

from swarms.structs.advisor_swarm import AdvisorSwarm
from swarms.structs.agent import Agent


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAdvisorSwarmInit:
    def test_default_construction(self):
        swarm = AdvisorSwarm()
        assert swarm.name == "AdvisorSwarm"
        assert swarm.executor_model_name == "claude-sonnet-4-6"
        assert swarm.advisor_model_name == "claude-opus-4-6"
        assert swarm.max_advisor_uses == 3
        assert swarm.max_loops == 1
        assert swarm.executor_agent is not None
        assert swarm.advisor_agent is not None
        assert swarm.conversation is not None

    def test_custom_model_names(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1",
        )
        assert swarm.executor_model_name == "gpt-4.1-mini"
        assert swarm.advisor_model_name == "gpt-4.1"
        assert swarm.executor_agent.model_name == "gpt-4.1-mini"
        assert swarm.advisor_agent.model_name == "gpt-4.1"

    def test_custom_agents_override(self):
        custom_executor = Agent(
            agent_name="CustomExecutor",
            model_name="gpt-4.1-mini",
            max_loops=1,
        )
        custom_advisor = Agent(
            agent_name="CustomAdvisor",
            model_name="gpt-4.1",
            max_loops=1,
        )
        swarm = AdvisorSwarm(
            executor_agent=custom_executor,
            advisor_agent=custom_advisor,
        )
        assert swarm.executor_agent is custom_executor
        assert swarm.advisor_agent is custom_advisor

    def test_id_generated(self):
        swarm = AdvisorSwarm()
        assert swarm.id is not None
        assert len(swarm.id) > 0

    def test_callable(self):
        swarm = AdvisorSwarm()
        assert callable(swarm)


# ---------------------------------------------------------------------------
# Reliability check tests
# ---------------------------------------------------------------------------


class TestReliabilityCheck:
    def test_max_advisor_uses_zero_raises(self):
        with pytest.raises(ValueError, match="max_advisor_uses"):
            AdvisorSwarm(max_advisor_uses=0)

    def test_max_advisor_uses_negative_raises(self):
        with pytest.raises(ValueError, match="max_advisor_uses"):
            AdvisorSwarm(max_advisor_uses=-1)

    def test_max_loops_zero_raises(self):
        with pytest.raises(ValueError, match="max_loops"):
            AdvisorSwarm(max_loops=0)

    def test_empty_executor_model_raises(self):
        with pytest.raises(ValueError, match="executor_model_name"):
            AdvisorSwarm(executor_model_name="")

    def test_empty_advisor_model_raises(self):
        with pytest.raises(ValueError, match="advisor_model_name"):
            AdvisorSwarm(advisor_model_name="")


# ---------------------------------------------------------------------------
# Verdict parsing tests
# ---------------------------------------------------------------------------


class TestIsSatisfactory:
    def setup_method(self):
        self.swarm = AdvisorSwarm()

    def test_satisfactory_exact(self):
        assert self.swarm._is_satisfactory("VERDICT: SATISFACTORY\nLooks good.")

    def test_satisfactory_lowercase(self):
        assert self.swarm._is_satisfactory("verdict: satisfactory\nAll criteria met.")

    def test_satisfactory_mixed_case(self):
        assert self.swarm._is_satisfactory("Verdict: Satisfactory\nWell done.")

    def test_needs_revision(self):
        assert not self.swarm._is_satisfactory("VERDICT: NEEDS_REVISION\n1. Fix the imports.")

    def test_no_verdict(self):
        assert not self.swarm._is_satisfactory("The output looks incomplete.")

    def test_empty_string(self):
        assert not self.swarm._is_satisfactory("")


# ---------------------------------------------------------------------------
# Run validation tests
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_run_no_task_raises(self):
        swarm = AdvisorSwarm()
        with pytest.raises(ValueError, match="task is required"):
            swarm.run(task=None)

    def test_run_empty_task_raises(self):
        swarm = AdvisorSwarm()
        with pytest.raises(ValueError, match="task is required"):
            swarm.run(task="")


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def setup_method(self):
        self.swarm = AdvisorSwarm()

    def test_planning_prompt_contains_task(self):
        prompt = self.swarm._build_planning_prompt("Write a haiku")
        assert "Write a haiku" in prompt
        assert "PLANNING MODE" in prompt

    def test_executor_prompt_contains_task_and_advice(self):
        prompt = self.swarm._build_executor_prompt(
            "Write a haiku", "1. Choose a theme\n2. Count syllables"
        )
        assert "Write a haiku" in prompt
        assert "Count syllables" in prompt

    def test_review_prompt_contains_output(self):
        prompt = self.swarm._build_review_prompt(
            "Write a haiku", "Old pond / frog jumps in / water sound"
        )
        assert "REVIEW MODE" in prompt
        assert "Old pond" in prompt
        assert "Write a haiku" in prompt

    def test_refinement_prompt_contains_all_context(self):
        prompt = self.swarm._build_refinement_prompt(
            "Write a haiku",
            "Old pond / frog jumps in / water sound",
            "VERDICT: NEEDS_REVISION\n1. Fix syllable count",
        )
        assert "Write a haiku" in prompt
        assert "Old pond" in prompt
        assert "Fix syllable count" in prompt


# ---------------------------------------------------------------------------
# Integration test (requires API key)
# ---------------------------------------------------------------------------


class TestAdvisorSwarmExecution:
    """End-to-end tests with real API calls.

    These tests require a valid LLM API key in the environment.
    Skip with: pytest -k 'not Execution'
    """

    def test_basic_run(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=2,
            max_loops=1,
        )
        result = swarm.run(task="What is 2 + 2? Answer in one word.")
        assert result is not None

        # Conversation should have entries from User, Advisor, and Executor
        history = swarm.conversation.to_dict()
        roles = [msg["role"] for msg in history]
        assert "User" in roles
        assert "Advisor" in roles
        assert "Executor" in roles

    def test_max_advisor_uses_one_no_review(self):
        """With max_advisor_uses=1, only the planning call happens — no review."""
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        result = swarm.run(task="Say hello")
        assert result is not None

        history = swarm.conversation.to_dict()
        advisor_entries = [
            msg for msg in history if msg["role"] == "Advisor"
        ]
        # Only 1 advisor entry (the planning call)
        assert len(advisor_entries) == 1

    def test_batched_run(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        results = swarm.batched_run(["Say hi", "Say bye"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_callable_invocation(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        result = swarm("What is 1 + 1?")
        assert result is not None
