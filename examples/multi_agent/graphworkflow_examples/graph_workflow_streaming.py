"""
GraphWorkflow Streaming Callback Example

Demonstrates the on_node_complete callback that fires in real-time
as each agent finishes, before the full workflow completes.

Architecture:
    Coordinator (Layer 0)
        -> Market-Analyst   (Layer 1, parallel)
        -> Tech-Analyst     (Layer 1, parallel)
        -> Risk-Analyst     (Layer 1, parallel)
            -> Synthesizer  (Layer 2)

Watch the terminal -- you'll see each agent's result stream in
as it completes, with timestamps showing the real-time behavior.
"""

import time

from swarms.structs.agent import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_agent(name: str, description: str) -> Agent:
    return Agent(
        agent_name=name,
        agent_description=description,
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


def main():
    # -- Build agents --
    coordinator = create_agent(
        "Coordinator",
        "You coordinate analysis tasks. Briefly outline what each team member should focus on.",
    )
    market_analyst = create_agent(
        "Market-Analyst",
        "You analyse market trends and competitive landscape.",
    )
    tech_analyst = create_agent(
        "Tech-Analyst",
        "You evaluate technical feasibility and architecture.",
    )
    risk_analyst = create_agent(
        "Risk-Analyst",
        "You identify risks and propose mitigations.",
    )
    synthesizer = create_agent(
        "Synthesizer",
        "You synthesize inputs from multiple analysts into a concise executive summary.",
    )

    # -- Build workflow --
    workflow = GraphWorkflow(name="Streaming-Demo")
    for agent in [coordinator, market_analyst, tech_analyst, risk_analyst, synthesizer]:
        workflow.add_node(agent)

    # Coordinator fans out to three parallel analysts
    workflow.add_edges_from_source(
        "Coordinator",
        ["Market-Analyst", "Tech-Analyst", "Risk-Analyst"],
    )
    # All three analysts converge into the synthesizer
    workflow.add_edges_to_target(
        ["Market-Analyst", "Tech-Analyst", "Risk-Analyst"],
        "Synthesizer",
    )

    # -- Define the streaming callback --
    start_time = time.time()

    def on_node_complete(node_id: str, output) -> None:
        elapsed = time.time() - start_time
        preview = str(output)[:120].replace("\n", " ")
        print(f"\n  [{elapsed:6.2f}s] {node_id} completed:")
        print(f"           {preview}...")
        print()

    # -- Run with streaming --
    task = "Evaluate the feasibility of launching an AI-powered personal finance assistant."

    print("=" * 60)
    print("  GraphWorkflow Streaming Demo")
    print("=" * 60)
    print(f"\n  Task: {task}\n")
    print("  Waiting for agents to complete...\n")

    result = workflow.run(task, on_node_complete=on_node_complete)

    total = time.time() - start_time
    print("=" * 60)
    print(f"  All agents done in {total:.2f}s")
    print(f"  Agents completed: {list(result.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
