import pytest

from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.agent import Agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name"""
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Respond with your name and the task.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_round_robin_swarm_initialization():
    """Test RoundRobinSwarm initialization"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, verbose=True, max_loops=2)

    assert isinstance(swarm, RoundRobinSwarm)
    assert swarm.verbose is True
    assert swarm.max_loops == 2
    assert len(swarm.agents) == 3


def test_round_robin_swarm_default_initialization():
    """Test RoundRobinSwarm with default parameters"""
    agents = [create_test_agent("Agent1")]
    swarm = RoundRobinSwarm(agents=agents)

    assert swarm.verbose is False
    assert swarm.max_loops == 1
    assert swarm.index == 0


def test_round_robin_swarm_with_multiple_agents():
    """Test RoundRobinSwarm with multiple agents"""
    agents = [create_test_agent(f"Agent{i}") for i in range(5)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    assert len(swarm.agents) == 5


# ============================================================================
# EXECUTION TESTS
# ============================================================================


def test_round_robin_swarm_run():
    """Test basic run functionality"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    task = "Test task"
    result = swarm.run(task)

    assert result is not None
    assert swarm.index == 0  # Should reset to 0 after full cycle


def test_round_robin_swarm_single_agent_run():
    """Test run with single agent"""
    agents = [create_test_agent("Agent1")]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    task = "Single agent task"
    result = swarm.run(task)

    assert result is not None


def test_round_robin_swarm_multiple_loops():
    """Test run with multiple loops"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=3)

    task = "Multi-loop task"
    result = swarm.run(task)

    assert result is not None


# ============================================================================
# INDEX ROTATION TESTS
# ============================================================================


def test_round_robin_swarm_index_rotation():
    """Test that index properly rotates through agents"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    initial_index = swarm.index
    swarm.run("Task 1")

    # After a full cycle with max_loops=1 and 3 agents, index should reset to 0
    assert swarm.index == initial_index


def test_round_robin_swarm_maintains_order():
    """Test that agents are called in round-robin order"""
    agents = [create_test_agent(f"Agent{i}") for i in range(4)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1, verbose=True)

    # Initial index should be 0
    assert swarm.index == 0

    task = "Order test task"
    swarm.run(task)

    # After running through all agents once, should be back at 0
    assert swarm.index == 0


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_round_robin_swarm_verbose_mode():
    """Test RoundRobinSwarm with verbose mode enabled"""
    agents = [create_test_agent(f"Agent{i}") for i in range(2)]
    swarm = RoundRobinSwarm(agents=agents, verbose=True, max_loops=1)

    assert swarm.verbose is True

    result = swarm.run("Verbose test task")
    assert result is not None


def test_round_robin_swarm_quiet_mode():
    """Test RoundRobinSwarm with verbose mode disabled"""
    agents = [create_test_agent(f"Agent{i}") for i in range(2)]
    swarm = RoundRobinSwarm(agents=agents, verbose=False, max_loops=1)

    assert swarm.verbose is False

    result = swarm.run("Quiet test task")
    assert result is not None


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


def test_round_robin_swarm_empty_task():
    """Test RoundRobinSwarm with empty task string"""
    agents = [create_test_agent("Agent1")]
    swarm = RoundRobinSwarm(agents=agents)

    # Should handle empty string gracefully
    try:
        result = swarm.run("")
        assert result is not None or result == ""
    except (ValueError, TypeError):
        # Also acceptable to raise an error
        pass


def test_round_robin_swarm_long_task():
    """Test RoundRobinSwarm with a longer task"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    long_task = "This is a much longer task description " * 10
    result = swarm.run(long_task)

    assert result is not None


# ============================================================================
# AGENT COUNT VARIATION TESTS
# ============================================================================


def test_round_robin_swarm_two_agents():
    """Test RoundRobinSwarm with two agents"""
    agents = [create_test_agent(f"Agent{i}") for i in range(2)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    result = swarm.run("Two agent task")
    assert result is not None


def test_round_robin_swarm_many_agents():
    """Test RoundRobinSwarm with many agents"""
    agents = [create_test_agent(f"Agent{i}") for i in range(10)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    assert len(swarm.agents) == 10

    result = swarm.run("Many agent task")
    assert result is not None


# ============================================================================
# LOOP VARIATION TESTS
# ============================================================================


def test_round_robin_swarm_single_loop():
    """Test RoundRobinSwarm with single loop"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    assert swarm.max_loops == 1

    result = swarm.run("Single loop task")
    assert result is not None


def test_round_robin_swarm_multiple_max_loops():
    """Test RoundRobinSwarm with different max_loops values"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]

    for max_loops in [1, 2, 5]:
        swarm = RoundRobinSwarm(agents=agents, max_loops=max_loops)
        assert swarm.max_loops == max_loops

        result = swarm.run(f"Task with {max_loops} loops")
        assert result is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_round_robin_swarm_multiple_sequential_runs():
    """Test running multiple tasks sequentially"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    tasks = ["Task 1", "Task 2", "Task 3"]

    for task in tasks:
        result = swarm.run(task)
        assert result is not None


def test_round_robin_swarm_state_persistence():
    """Test that swarm maintains state across runs"""
    agents = [create_test_agent(f"Agent{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1)

    initial_agent_count = len(swarm.agents)

    swarm.run("First task")
    swarm.run("Second task")

    # Agent count should remain the same
    assert len(swarm.agents) == initial_agent_count

if __name__ == "__main__":
    pytest.main([__file__, "-v"])