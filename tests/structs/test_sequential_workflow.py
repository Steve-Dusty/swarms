import os
from unittest.mock import MagicMock, patch

import pytest

from swarms import Agent, SequentialWorkflow
from swarms.structs.sequential_workflow import (
    DriftDetectionAgent,
    DriftDetectionError,
    DriftDetectionResult,
    _WorkflowResult,
)
from swarms.utils.workspace_utils import get_workspace_dir


def test_sequential_workflow_initialization_with_agents():
    """Test SequentialWorkflow initialization with agents"""
    agent1 = Agent(
        agent_name="Agent-1",
        agent_description="First test agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Agent-2",
        agent_description="Second test agent",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Test-Workflow",
        description="Test workflow with multiple agents",
        agents=[agent1, agent2],
        max_loops=1,
    )

    assert isinstance(workflow, SequentialWorkflow)
    assert workflow.name == "Test-Workflow"
    assert (
        workflow.description == "Test workflow with multiple agents"
    )
    assert len(workflow.agents) == 2
    assert workflow.agents[0] == agent1
    assert workflow.agents[1] == agent2
    assert workflow.max_loops == 1


def test_sequential_workflow_multi_agent_execution():
    """Test SequentialWorkflow execution with multiple agents"""
    agent1 = Agent(
        agent_name="Research-Agent",
        agent_description="Agent for research tasks",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Analysis-Agent",
        agent_description="Agent for analyzing research results",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Summary-Agent",
        agent_description="Agent for summarizing findings",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Multi-Agent-Research-Workflow",
        description="Workflow for comprehensive research, analysis, and summarization",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test that the workflow executes successfully
    result = workflow.run(
        "Analyze the impact of renewable energy on climate change"
    )
    assert result is not None
    # SequentialWorkflow may return different types based on output_type, just ensure it's not None


def test_sequential_workflow_batched_execution():
    """Test batched execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Data-Collector",
        agent_description="Agent for collecting data",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Data-Processor",
        agent_description="Agent for processing collected data",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Batched-Processing-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
    )

    # Test batched execution
    tasks = [
        "Analyze solar energy trends",
        "Evaluate wind power efficiency",
        "Compare renewable energy sources",
    ]
    results = workflow.run_batched(tasks)
    assert results is not None
    # run_batched returns a list of results
    assert isinstance(results, list)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_sequential_workflow_async_execution():
    """Test async execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Async-Research-Agent",
        agent_description="Agent for async research tasks",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Async-Analysis-Agent",
        agent_description="Agent for async analysis",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Async-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
    )

    # Test async execution
    result = await workflow.run_async("Analyze AI trends in 2024")
    assert result is not None


@pytest.mark.asyncio
async def test_sequential_workflow_concurrent_execution():
    """Test concurrent execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Concurrent-Research-Agent",
        agent_description="Agent for concurrent research",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Concurrent-Analysis-Agent",
        agent_description="Agent for concurrent analysis",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Concurrent-Summary-Agent",
        agent_description="Agent for concurrent summarization",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Concurrent-Workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test concurrent execution
    tasks = [
        "Research quantum computing advances",
        "Analyze blockchain technology trends",
        "Evaluate machine learning applications",
    ]
    results = await workflow.run_concurrent(tasks)
    assert results is not None
    # run_concurrent returns a list of results
    assert isinstance(results, list)
    assert len(results) == 3


def test_sequential_workflow_with_multi_agent_collaboration():
    """Test SequentialWorkflow with multi-agent collaboration prompts"""
    agent1 = Agent(
        agent_name="Market-Research-Agent",
        agent_description="Agent for market research",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Competitive-Analysis-Agent",
        agent_description="Agent for competitive analysis",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Strategy-Development-Agent",
        agent_description="Agent for developing business strategies",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Business-Strategy-Workflow",
        description="Comprehensive business strategy development workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
        multi_agent_collab_prompt=True,
    )

    # Test that collaboration prompt is added
    assert agent1.system_prompt is not None
    assert agent2.system_prompt is not None
    assert agent3.system_prompt is not None

    # Test execution
    result = workflow.run(
        "Develop a business strategy for entering the AI market"
    )
    assert result is not None


def test_sequential_workflow_error_handling():
    """Test SequentialWorkflow error handling"""
    # Test with invalid agents list
    with pytest.raises(
        ValueError, match="Agents list cannot be None or empty"
    ):
        SequentialWorkflow(agents=None)

    with pytest.raises(
        ValueError, match="Agents list cannot be None or empty"
    ):
        SequentialWorkflow(agents=[])

    # Test with zero max_loops
    with pytest.raises(ValueError, match="max_loops cannot be 0"):
        agent1 = Agent(
            agent_name="Test-Agent",
            agent_description="Test agent",
            model_name="gpt-4o",
            max_loops=1,
        )
        SequentialWorkflow(agents=[agent1], max_loops=0)


def test_sequential_workflow_agent_names_extraction():
    """Test that SequentialWorkflow properly extracts agent names for flow"""
    agent1 = Agent(
        agent_name="Alpha-Agent",
        agent_description="First agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Beta-Agent",
        agent_description="Second agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Gamma-Agent",
        agent_description="Third agent",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Test-Flow-Workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test flow string generation
    expected_flow = "Alpha-Agent -> Beta-Agent -> Gamma-Agent"
    assert workflow.flow == expected_flow


def test_sequential_workflow_team_awareness():
    """Test SequentialWorkflow with team awareness enabled"""
    agent1 = Agent(
        agent_name="Team-Member-1",
        agent_description="First team member",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Team-Member-2",
        agent_description="Second team member",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Team-Aware-Workflow",
        description="Workflow with team awareness",
        agents=[agent1, agent2],
        max_loops=1,
        team_awareness=True,
    )

    # Test that workflow initializes successfully with team awareness
    assert workflow.team_awareness is True
    assert len(workflow.agents) == 2


def test_sequential_workflow_autosave_creates_workspace_dir(
    monkeypatch, tmp_path
):
    """Test that SequentialWorkflow with autosave=True creates a workspace directory."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Agent-1",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Autosave-Agent-2",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Autosave-Test-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
    )

    assert workflow.autosave is True
    assert workflow.swarm_workspace_dir is not None
    assert os.path.isdir(workflow.swarm_workspace_dir)
    assert "SequentialWorkflow" in workflow.swarm_workspace_dir
    assert "Autosave-Test-Workflow" in workflow.swarm_workspace_dir

    get_workspace_dir.cache_clear()


def test_sequential_workflow_autosave_saves_conversation_after_run(
    monkeypatch, tmp_path
):
    """Test that SequentialWorkflow saves conversation_history.json after run when autosave=True."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Run-Agent-1",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Run-Agent-2",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    workflow = SequentialWorkflow(
        name="Autosave-Run-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    result = workflow.run("Say hello in one short sentence.")
    assert result is not None

    conversation_path = os.path.join(
        workflow.swarm_workspace_dir, "conversation_history.json"
    )
    assert os.path.isfile(
        conversation_path
    ), f"Expected conversation_history.json at {conversation_path}"

    get_workspace_dir.cache_clear()


# ---------------------------------------------------------------------------
# DriftDetectionResult
# ---------------------------------------------------------------------------


def test_drift_detection_result_fields():
    result = DriftDetectionResult(
        score=0.9,
        status="ok",
        output="some output",
        original_task="task",
    )
    assert result.score == 0.9
    assert result.status == "ok"
    assert result.output == "some output"
    assert result.original_task == "task"


def test_drift_detection_result_default_original_task():
    result = DriftDetectionResult(
        score=0.5, status="drift_detected", output="x"
    )
    assert result.original_task == ""


# ---------------------------------------------------------------------------
# DriftDetectionError
# ---------------------------------------------------------------------------


def test_drift_detection_error_is_exception():
    with pytest.raises(DriftDetectionError):
        raise DriftDetectionError("drift too high")


# ---------------------------------------------------------------------------
# _WorkflowResult
# ---------------------------------------------------------------------------


def test_workflow_result_str():
    dr = DriftDetectionResult(score=0.9, status="ok", output="hello")
    wr = _WorkflowResult(output="hello", drift=dr)
    assert str(wr) == "hello"
    assert wr.drift is dr


# ---------------------------------------------------------------------------
# DriftDetectionAgent.score
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str):
    """Build a minimal mock that looks like a litellm completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_score_returns_float_from_llm():
    agent = DriftDetectionAgent(judge_model="gpt-4o")
    with patch(
        "litellm.completion",
        return_value=_make_litellm_response("0.85"),
    ):
        score = agent.score("task", "output")
    assert score == pytest.approx(0.85)


def test_score_clamps_above_one():
    agent = DriftDetectionAgent()
    with patch(
        "litellm.completion",
        return_value=_make_litellm_response("1.5"),
    ):
        score = agent.score("task", "output")
    assert score == 1.0


def test_score_clamps_below_zero():
    agent = DriftDetectionAgent()
    with patch(
        "litellm.completion",
        return_value=_make_litellm_response("-0.3"),
    ):
        score = agent.score("task", "output")
    assert score == 0.0


def test_score_defaults_to_one_on_llm_failure():
    agent = DriftDetectionAgent()
    with patch(
        "litellm.completion", side_effect=RuntimeError("API error")
    ):
        score = agent.score("task", "output")
    assert score == 1.0


def test_score_defaults_to_one_on_parse_failure():
    agent = DriftDetectionAgent()
    with patch(
        "litellm.completion",
        return_value=_make_litellm_response("not-a-number"),
    ):
        score = agent.score("task", "output")
    assert score == 1.0


# ---------------------------------------------------------------------------
# DriftDetectionAgent.run — on_drift="flag"
# ---------------------------------------------------------------------------


def test_run_flag_ok_when_score_above_threshold():
    agent = DriftDetectionAgent(threshold=0.75, on_drift="flag")
    with patch.object(agent, "score", return_value=0.9):
        result = agent.run(
            "task", "output", pipeline_runner=lambda t: "unused"
        )
    assert result.status == "ok"
    assert result.score == pytest.approx(0.9)
    assert result.output == "output"
    assert result.original_task == "task"


def test_run_flag_drift_detected_when_score_below_threshold():
    agent = DriftDetectionAgent(threshold=0.75, on_drift="flag")
    with patch.object(agent, "score", return_value=0.5):
        result = agent.run(
            "task", "output", pipeline_runner=lambda t: "unused"
        )
    assert result.status == "drift_detected"
    assert result.score == pytest.approx(0.5)


def test_run_flag_exact_threshold_is_ok():
    agent = DriftDetectionAgent(threshold=0.75, on_drift="flag")
    with patch.object(agent, "score", return_value=0.75):
        result = agent.run(
            "task", "output", pipeline_runner=lambda t: "unused"
        )
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# DriftDetectionAgent.run — on_drift="raise"
# ---------------------------------------------------------------------------


def test_run_raise_raises_on_drift():
    agent = DriftDetectionAgent(threshold=0.75, on_drift="raise")
    with patch.object(agent, "score", return_value=0.4):
        with pytest.raises(DriftDetectionError, match="score=0.40"):
            agent.run(
                "task", "output", pipeline_runner=lambda t: "unused"
            )


def test_run_raise_does_not_raise_when_ok():
    agent = DriftDetectionAgent(threshold=0.75, on_drift="raise")
    with patch.object(agent, "score", return_value=0.9):
        result = agent.run(
            "task", "output", pipeline_runner=lambda t: "unused"
        )
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# DriftDetectionAgent.run — on_drift="rerun"
# ---------------------------------------------------------------------------


def test_run_rerun_succeeds_on_second_attempt():
    agent = DriftDetectionAgent(
        threshold=0.75, on_drift="rerun", max_retries=2
    )
    scores = iter([0.4, 0.9])
    runner = MagicMock(return_value="new output")
    with patch.object(agent, "score", side_effect=scores):
        result = agent.run(
            "task", "old output", pipeline_runner=runner
        )
    assert result.status == "rerun_complete"
    assert result.output == "new output"
    assert result.score == pytest.approx(0.9)
    runner.assert_called_once_with("task")


def test_run_rerun_exhausts_retries_returns_drift_detected():
    # After exhausting retries without meeting threshold, status must be
    # "drift_detected" (falls back to flag behavior), NOT "rerun_complete".
    agent = DriftDetectionAgent(
        threshold=0.75, on_drift="rerun", max_retries=2
    )
    scores = iter([0.3, 0.4, 0.5])
    runner = MagicMock(return_value="retried output")
    with patch.object(agent, "score", side_effect=scores):
        result = agent.run(
            "task", "old output", pipeline_runner=runner
        )
    assert result.status == "drift_detected"
    assert runner.call_count == 2  # max_retries=2


def test_run_rerun_complete_only_when_threshold_met():
    # "rerun_complete" must only appear when the rerun actually meets the threshold.
    agent = DriftDetectionAgent(
        threshold=0.75, on_drift="rerun", max_retries=3
    )
    scores = iter([0.3, 0.5, 0.9])
    runner = MagicMock(return_value="better output")
    with patch.object(agent, "score", side_effect=scores):
        result = agent.run(
            "task", "old output", pipeline_runner=runner
        )
    assert result.status == "rerun_complete"
    assert result.score == pytest.approx(0.9)


def test_run_rerun_respects_max_retries_of_one():
    agent = DriftDetectionAgent(
        threshold=0.75, on_drift="rerun", max_retries=1
    )
    scores = iter([0.2, 0.3])
    runner = MagicMock(return_value="retried")
    with patch.object(agent, "score", side_effect=scores):
        agent.run("task", "original", pipeline_runner=runner)
    assert runner.call_count == 1


# ---------------------------------------------------------------------------
# SequentialWorkflow drift_detection integration
# ---------------------------------------------------------------------------


def _make_workflow(drift_detection=False):
    a1 = Agent(agent_name="A1", model_name="gpt-4o", max_loops=1)
    a2 = Agent(agent_name="A2", model_name="gpt-4o", max_loops=1)
    return SequentialWorkflow(
        agents=[a1, a2],
        max_loops=1,
        autosave=False,
        drift_detection=drift_detection,
    )


def test_workflow_drift_detection_false_stores_none():
    wf = _make_workflow(drift_detection=False)
    assert wf.drift_detection is None


def test_workflow_drift_detection_true_creates_agent():
    wf = _make_workflow(drift_detection=True)
    assert isinstance(wf.drift_detection, DriftDetectionAgent)


def test_workflow_drift_detection_accepts_preconfigured_agent():
    agent = DriftDetectionAgent(
        threshold=0.85, on_drift="rerun", max_retries=3
    )
    a1 = Agent(agent_name="A1", model_name="gpt-4o", max_loops=1)
    a2 = Agent(agent_name="A2", model_name="gpt-4o", max_loops=1)
    wf = SequentialWorkflow(
        agents=[a1, a2],
        max_loops=1,
        autosave=False,
        drift_detection=agent,
    )
    assert wf.drift_detection is agent
    assert wf.drift_detection.threshold == 0.85
    assert wf.drift_detection.on_drift == "rerun"
    assert wf.drift_detection.max_retries == 3


def test_workflow_run_with_drift_detection_attaches_drift_attr():
    wf = _make_workflow(drift_detection=True)
    fake_output = "pipeline result"
    with patch.object(
        wf.agent_rearrange, "run", return_value=fake_output
    ), patch.object(wf.drift_detection, "score", return_value=0.95):
        result = wf.run("do something")
    assert hasattr(result, "drift")
    assert isinstance(result.drift, DriftDetectionResult)
    assert result.drift.status == "ok"
    assert result.drift.score == pytest.approx(0.95)


def test_workflow_run_without_drift_detection_returns_raw():
    wf = _make_workflow(drift_detection=False)
    fake_output = "pipeline result"
    with patch.object(
        wf.agent_rearrange, "run", return_value=fake_output
    ):
        result = wf.run("do something")
    assert result == fake_output
    assert not hasattr(result, "drift")


def test_workflow_run_drift_detected_status():
    wf = _make_workflow(drift_detection=True)
    with patch.object(
        wf.agent_rearrange, "run", return_value="output"
    ), patch.object(wf.drift_detection, "score", return_value=0.3):
        result = wf.run("task")
    assert result.drift.status == "drift_detected"


def test_workflow_run_rerun_uses_new_output():
    wf = _make_workflow(drift_detection=True)
    wf.drift_detection.on_drift = "rerun"
    wf.drift_detection.max_retries = 1
    run_outputs = iter(["old output", "new output"])
    scores = iter([0.2, 0.9])
    with patch.object(
        wf.agent_rearrange, "run", side_effect=run_outputs
    ), patch.object(wf.drift_detection, "score", side_effect=scores):
        result = wf.run("task")
    assert str(result) == "new output"
    assert result.drift.status == "rerun_complete"


def test_workflow_run_raise_propagates_error():
    wf = _make_workflow(drift_detection=True)
    wf.drift_detection.on_drift = "raise"
    with patch.object(
        wf.agent_rearrange, "run", return_value="output"
    ), patch.object(wf.drift_detection, "score", return_value=0.1):
        with pytest.raises(DriftDetectionError):
            wf.run("task")


def test_workflow_run_drift_with_dict_output_proxies_access():
    # Default output_type="dict" means agent_rearrange.run may return a dict.
    # Enabling drift_detection must not break dict access patterns.
    wf = _make_workflow(drift_detection=True)
    fake_output = {"Agent1": "research notes", "Agent2": "analysis"}
    with patch.object(
        wf.agent_rearrange, "run", return_value=fake_output
    ), patch.object(wf.drift_detection, "score", return_value=0.9):
        result = wf.run("task")
    assert hasattr(result, "drift")
    assert result.drift.status == "ok"
    # Full mapping contract must be preserved
    assert isinstance(result, dict)
    assert result["Agent1"] == "research notes"
    assert result["Agent2"] == "analysis"
    assert "Agent1" in result
    assert len(result) == 2
    assert set(result) == {"Agent1", "Agent2"}
    assert set(result.keys()) == {"Agent1", "Agent2"}
    assert result.get("Agent1") == "research notes"
    assert result.get("missing", "default") == "default"
    assert dict(result.items()) == fake_output
