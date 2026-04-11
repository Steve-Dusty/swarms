"""
Example: SequentialWorkflow with Drift Detection

Demonstrates how to enable drift_detection=True on a SequentialWorkflow so that
a DriftDetectionAgent automatically scores the final output's semantic alignment
with the original task after the pipeline completes.

The drift result is surfaced on the returned object as `.drift`:
    result.drift.score   — float in [0.0, 1.0]
    result.drift.status  — "ok" | "drift_detected" | "rerun_complete"
    result.drift.output  — the final output (original or from a re-run)
"""

from swarms import Agent, SequentialWorkflow
from swarms.structs.sequential_workflow import DriftDetectionAgent, DriftDetectionError

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = Agent(
    agent_name="Researcher",
    system_prompt="""You are a research specialist. Given a topic, produce a
    concise factual summary covering the key points, major actors, and
    recent developments.""",
    model_name="gpt-4o",
    max_loops=1,
    temperature=0.3,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="""You are an analytical expert. Given research notes,
    identify the most significant implications and surface three clear
    takeaways that directly address the original question.""",
    model_name="gpt-4o",
    max_loops=1,
    temperature=0.3,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="""You are a professional writer. Given analytical takeaways,
    produce a polished, reader-friendly summary in 2-3 paragraphs that
    directly answers the original question.""",
    model_name="gpt-4o",
    max_loops=1,
    temperature=0.5,
)

# ---------------------------------------------------------------------------
# Example 1 — drift_detection=True (default DriftDetectionAgent settings)
# threshold=0.75, on_drift="flag", judge_model="gpt-4o"
# ---------------------------------------------------------------------------

wf_flag = SequentialWorkflow(
    name="geopolitics-pipeline",
    agents=[researcher, analyst, writer],
    max_loops=1,
    drift_detection=True,
)

task = "Summarize the geopolitical impact of rare earth mining in the Congo"

result = wf_flag.run(task)

print("=== Example 1: flag mode ===")
print(f"Output:\n{result}\n")
print(f"Drift score : {result.drift.score:.2f}")
print(f"Drift status: {result.drift.status}")

# ---------------------------------------------------------------------------
# Example 2 — custom DriftDetectionAgent with rerun on drift
# Pass the agent directly to drift_detection after constructing it manually
# and assigning it post-init (drift_detection param accepts bool only, so we
# swap the attribute after construction for custom configuration).
# ---------------------------------------------------------------------------

wf_rerun = SequentialWorkflow(
    name="geopolitics-pipeline-rerun",
    agents=[researcher, analyst, writer],
    max_loops=1,
    drift_detection=True,
)

# Override with a custom agent for finer control
wf_rerun.drift_detection = DriftDetectionAgent(
    threshold=0.80,
    on_drift="rerun",
    max_retries=2,
    judge_model="gpt-4o",
)

result2 = wf_rerun.run(task)

print("\n=== Example 2: rerun mode ===")
print(f"Output:\n{result2}\n")
print(f"Drift score : {result2.drift.score:.2f}")
print(f"Drift status: {result2.drift.status}")

# ---------------------------------------------------------------------------
# Example 3 — raise on drift (catch the error gracefully)
# ---------------------------------------------------------------------------

wf_raise = SequentialWorkflow(
    name="geopolitics-pipeline-strict",
    agents=[researcher, analyst, writer],
    max_loops=1,
    drift_detection=True,
)

wf_raise.drift_detection = DriftDetectionAgent(
    threshold=0.99,  # intentionally very strict to demonstrate raising
    on_drift="raise",
    judge_model="gpt-4o",
)

print("\n=== Example 3: raise mode (strict threshold) ===")
try:
    result3 = wf_raise.run(task)
    print(f"Output:\n{result3}")
    print(f"Drift score : {result3.drift.score:.2f}")
except DriftDetectionError as exc:
    print(f"Caught DriftDetectionError: {exc}")
