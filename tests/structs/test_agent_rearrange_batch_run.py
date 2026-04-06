"""
Unit tests for AgentRearrange.batch_run() concurrent execution.

Tests verify:
1. batch_run uses ThreadPoolExecutor (not sequential) within each batch
2. Each task gets its own conversation copy — no shared-state corruption
3. Result ordering is preserved across all tasks and batch boundaries
4. Works correctly with and without image paths
5. batch_size boundary conditions are handled correctly
6. Fallback path (non-deepcopyable agents) is safe via _LockedAgent proxy

Note: validate_flow() requires '->' in the flow, so all pipelines here use
at least two agents.
"""

import copy
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.conversation import Conversation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, delay: float = 0.0):
    """Return a minimal mock Agent that sleeps *delay* seconds then echoes."""
    agent = MagicMock()
    agent.agent_name = name
    agent.system_prompt = f"I am {name}."

    def _run(task, *a, **kw):
        if delay:
            time.sleep(delay)
        return f"{name}:{task}"

    agent.run = _run
    return agent


def _make_pipeline(
    *agent_names: str, delay: float = 0.0
) -> AgentRearrange:
    """
    Build a sequential AgentRearrange pipeline with mock agents.

    Requires at least 2 agent names because validate_flow() demands '->'.
    """
    assert len(agent_names) >= 2, "_make_pipeline needs ≥2 agents"
    agents = [_make_agent(n, delay=delay) for n in agent_names]
    flow = " -> ".join(agent_names)
    return AgentRearrange(
        agents=agents,
        flow=flow,
        max_loops=1,
        autosave=False,
        output_type="final",
    )


# ---------------------------------------------------------------------------
# 1. Return-value correctness
# ---------------------------------------------------------------------------


class TestBatchRunReturns:
    def test_returns_list_same_length_as_tasks(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]
        results = pipeline.batch_run(tasks=tasks, batch_size=10)
        assert isinstance(results, list)
        assert len(results) == len(tasks)

    def test_single_task(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(
            tasks=["only task"], batch_size=5
        )
        assert len(results) == 1

    def test_empty_task_list(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(tasks=[], batch_size=5)
        assert results == []

    def test_results_in_input_order(self):
        """Results must match the task order, not thread-completion order."""
        pipeline = _make_pipeline("AgentA", "AgentB", delay=0.02)
        tasks = [f"task-{i}" for i in range(8)]
        results = pipeline.batch_run(tasks=tasks, batch_size=8)
        for i, result in enumerate(results):
            assert (
                f"task-{i}" in result
            ), f"result[{i}] = {result!r} does not contain task-{i}"

    def test_results_across_multiple_batches_ordered(self):
        """Order must be preserved even when tasks span multiple batches."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"item-{i}" for i in range(7)]
        results = pipeline.batch_run(tasks=tasks, batch_size=3)
        assert len(results) == 7
        for i, result in enumerate(results):
            assert f"item-{i}" in result


# ---------------------------------------------------------------------------
# 2. Concurrency — tasks within a batch run in parallel
# ---------------------------------------------------------------------------


class TestBatchRunConcurrency:
    def test_batch_is_faster_than_sequential(self):
        """
        With N tasks each taking ~T seconds, concurrent execution should
        complete in significantly less than N*T seconds.
        """
        N = 5
        T = 0.1  # seconds per task
        pipeline = _make_pipeline("SlowAgent", "SlowAgent2", delay=T)
        tasks = [f"task-{i}" for i in range(N)]

        t0 = time.perf_counter()
        results = pipeline.batch_run(tasks=tasks, batch_size=N)
        elapsed = time.perf_counter() - t0

        # Sequential would take ~N*T; concurrent should be well under that
        sequential_time = N * T
        assert elapsed < sequential_time * 0.7, (
            f"Expected elapsed < {sequential_time * 0.7:.2f}s "
            f"(70% of sequential {sequential_time:.2f}s), got {elapsed:.2f}s"
        )
        assert len(results) == N

    def test_threadpoolexecutor_is_used(self):
        """Patch ThreadPoolExecutor to confirm it is invoked for each batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=10)
            # One batch → ThreadPoolExecutor called once
            assert mock_tpe.call_count == 1

    def test_multiple_batches_uses_executor_per_batch(self):
        """One ThreadPoolExecutor context-manager per batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"t{i}" for i in range(6)]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=2)
            # 6 tasks / batch_size=2 → 3 batches → 3 executor instances
            assert mock_tpe.call_count == 3


# ---------------------------------------------------------------------------
# 3. Conversation + agent-state isolation — no shared state between tasks
# ---------------------------------------------------------------------------


class TestConversationIsolation:
    def test_deepcopy_called_per_task(self):
        """
        batch_run must call copy.deepcopy once per task so both the
        conversation AND agent objects are isolated.
        """
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["alpha", "beta", "gamma"]

        deepcopy_calls: List[object] = []
        original_deepcopy = copy.deepcopy

        def _spy_deepcopy(obj):
            result = original_deepcopy(obj)
            # We care about cloning the full AgentRearrange (not sub-copies
            # that deepcopy triggers internally)
            from swarms.structs.agent_rearrange import AgentRearrange

            if isinstance(obj, AgentRearrange):
                deepcopy_calls.append(obj)
            return result

        with patch(
            "swarms.structs.agent_rearrange.copy.deepcopy",
            side_effect=_spy_deepcopy,
        ):
            pipeline.batch_run(tasks=tasks, batch_size=10)

        assert len(deepcopy_calls) == len(tasks), (
            f"Expected {len(tasks)} deepcopy calls on AgentRearrange, "
            f"got {len(deepcopy_calls)}"
        )

    def test_original_conversation_not_mutated(self):
        """The pipeline's own conversation should not change after batch_run."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        original_msg_count = len(
            pipeline.conversation.conversation_history
        )

        pipeline.batch_run(tasks=["task1", "task2"], batch_size=5)

        after_msg_count = len(
            pipeline.conversation.conversation_history
        )
        assert (
            after_msg_count == original_msg_count
        ), "pipeline.conversation was mutated by batch_run"

    def test_stateful_agent_state_does_not_bleed_across_tasks(self):
        """
        Regression: if agent objects were shared between concurrent workers,
        a stateful agent's last_task attribute would be overwritten by a
        racing thread, causing some results to contain the wrong task payload.

        This test verifies that each worker operates on its own deep-copied
        agent instances (no threading.Lock so the agent is deep-copyable),
        so no cross-task contamination occurs.
        """

        class StatefulAgent:
            """Agent that records which task it last processed.

            Deliberately lock-free — with deepcopy each task owns its own
            instance, so no synchronisation is needed.  The test would be
            unreliable (wrong task in result) if agents were shared.
            """

            def __init__(self, name: str):
                self.agent_name = name
                self.system_prompt = ""
                self.last_task: Optional[str] = None

            def run(self, task, *args, **kwargs):
                time.sleep(0.05)
                self.last_task = task
                return f"{self.agent_name} saw::{self.last_task}"

        a1 = StatefulAgent("Stage1")
        a2 = StatefulAgent("Stage2")

        pipeline = AgentRearrange(
            agents=[a1, a2],
            flow="Stage1 -> Stage2",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        N = 8
        tasks = [f"task-{i}" for i in range(N)]
        results = pipeline.batch_run(tasks=tasks, batch_size=N)

        assert len(results) == N
        for i, result in enumerate(results):
            assert f"task-{i}" in str(result), (
                f"result[{i}] does not contain 'task-{i}': {result!r}\n"
                "Likely cause: agent state was shared across concurrent workers."
            )

    def test_non_deepcopyable_agents_are_safe_via_locked_proxy(self):
        """
        Regression for the fallback path: when copy.deepcopy(self) fails
        because an agent contains a threading.Lock, batch_run must still
        produce correct, uncontaminated results by wrapping that agent in
        a _LockedAgent proxy that serialises concurrent .run() calls.

        Without the fix the old fallback left agents shared, so concurrent
        tasks overwrote each other's last_task and results were wrong.
        """

        class NonCopyableAgent:
            """Agent that cannot be deep-copied (holds a threading.Lock)."""

            def __init__(self, name: str):
                self.agent_name = name
                self.system_prompt = ""
                self._lock = threading.Lock()  # blocks deepcopy
                self.last_task: Optional[str] = None

            def run(self, task, *args, **kwargs):
                # Simulate meaningful work time to maximise race chance
                time.sleep(0.05)
                self.last_task = task
                return f"{self.agent_name} saw::{self.last_task}"

        a1 = NonCopyableAgent("Stage1")
        a2 = NonCopyableAgent("Stage2")

        pipeline = AgentRearrange(
            agents=[a1, a2],
            flow="Stage1 -> Stage2",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        N = 10
        tasks = [f"task-{i}" for i in range(N)]

        # Run multiple times to increase chance of catching a race
        for run_idx in range(3):
            results = pipeline.batch_run(tasks=tasks, batch_size=N)
            assert (
                len(results) == N
            ), f"run {run_idx}: wrong result count"
            for i, result in enumerate(results):
                assert f"task-{i}" in str(result), (
                    f"run {run_idx}, result[{i}] does not contain 'task-{i}': "
                    f"{result!r}\n"
                    "Agent state was shared — _LockedAgent proxy not working."
                )


# ---------------------------------------------------------------------------
# 4. Image paths forwarded correctly
# ---------------------------------------------------------------------------


class TestBatchRunWithImages:
    def test_img_list_passed_per_task(self):
        """When img is provided, each task receives its corresponding image path."""
        received: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        tasks = ["t1", "t2", "t3"]
        images = ["img1.png", "img2.png", "img3.png"]

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            results = pipeline.batch_run(
                tasks=tasks, img=images, batch_size=5
            )

        assert len(results) == 3
        # All images must have been forwarded (order may vary due to threading)
        assert sorted(received) == sorted(images)

    def test_no_img_passes_none(self):
        """When img is omitted, None is passed as img for every task."""
        received_imgs: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received_imgs.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            pipeline.batch_run(tasks=["t1", "t2"], batch_size=5)

        assert all(img is None for img in received_imgs)


# ---------------------------------------------------------------------------
# 5. batch_size boundary conditions
# ---------------------------------------------------------------------------


class TestBatchSizeBoundaries:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 100])
    def test_various_batch_sizes_return_all_results(self, batch_size):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"task-{i}" for i in range(5)]
        results = pipeline.batch_run(
            tasks=tasks, batch_size=batch_size
        )
        assert len(results) == len(tasks)

    def test_batch_size_one_still_concurrent_api(self):
        """Even batch_size=1 should go through ThreadPoolExecutor."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["only"]
        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=1)
            assert mock_tpe.call_count == 1
