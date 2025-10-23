import asyncio
import pytest
from typing import List

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    exponential_swarm,
    geometric_swarm,
    grid_swarm,
    harmonic_swarm,
    linear_swarm,
    log_swarm,
    mesh_swarm,
    one_to_one,
    one_to_three,
    power_swarm,
    pyramid_swarm,
    sigmoid_swarm,
    sinusoidal_swarm,
    staircase_swarm,
    star_swarm,
)

# ============================================================================
# Helper Functions
# ============================================================================


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name"""
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Respond with your name and the task you received.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


def create_test_agents(num_agents: int) -> List[Agent]:
    """Create specified number of test agents"""
    return [
        create_test_agent(f"Agent{i+1}") for i in range(num_agents)
    ]

# ============================================================================
# Basic Swarm Architecture Tests
# ============================================================================


def test_circular_swarm():
    """Test circular swarm functionality"""
    agents = create_test_agents(3)
    tasks = [
        "Analyze data",
        "Generate report", 
        "Summarize findings",
    ]

    result = circular_swarm(agents, tasks)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)
        assert len(log["response"]) > 0


def test_grid_swarm():
    """Test grid swarm functionality"""
    agents = create_test_agents(4)  # 2x2 grid
    tasks = ["Task A", "Task B", "Task C", "Task D"]

    result = grid_swarm(agents, tasks)
    
    assert result is not None
    assert isinstance(result, (dict, list))


def test_linear_swarm():
    """Test linear swarm functionality"""
    agents = create_test_agents(3)
    tasks = ["Research task", "Write content", "Review output"]

    result = linear_swarm(agents, tasks)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)


def test_star_swarm():
    """Test star swarm functionality"""
    agents = create_test_agents(4)  # 1 center + 3 peripheral
    tasks = ["Coordinate workflow", "Process data"]

    result = star_swarm(agents, tasks)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)


def test_mesh_swarm():
    """Test mesh swarm functionality"""
    agents = create_test_agents(3)
    tasks = [
        "Analyze data",
        "Process information", 
        "Generate insights",
    ]

    result = mesh_swarm(agents, tasks)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)


def test_pyramid_swarm():
    """Test pyramid swarm functionality"""
    agents = create_test_agents(6)  # 1-2-3 pyramid
    tasks = [
        "Top task",
        "Middle task 1",
        "Middle task 2", 
        "Bottom task 1",
        "Bottom task 2",
        "Bottom task 3",
    ]

    result = pyramid_swarm(agents, tasks)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)


# ============================================================================
# Communication Pattern Tests
# ============================================================================


def test_one_to_one_communication():
    """Test one-to-one communication pattern"""
    sender = create_test_agent("Sender")
    receiver = create_test_agent("Receiver")
    task = "Process and relay this message"

    result = one_to_one(sender, receiver, task)
    
    assert result is not None
    assert "history" in result
    assert len(result["history"]) > 0
    
    # Verify each log entry has required fields
    for log in result["history"]:
        assert "agent_name" in log
        assert "task" in log
        assert "response" in log
        assert isinstance(log["response"], str)


@pytest.mark.asyncio
async def test_one_to_three_communication():
    """Test one-to-three communication pattern"""
    sender = create_test_agent("Sender")
    receivers = create_test_agents(3)
    task = "Process and relay this message"

    result = await one_to_three(sender, receivers, task)
    
    assert result is not None


@pytest.mark.asyncio
async def test_broadcast_communication():
    """Test broadcast communication pattern"""
    sender = create_test_agent("Sender")
    broadcast_receivers = create_test_agents(5)
    task = "Process and relay this message"

    result = await broadcast(sender, broadcast_receivers, task)
    
    assert result is not None


# ============================================================================
# Mathematical Swarm Tests
# ============================================================================


def test_power_swarm():
    """Test power swarm functionality"""
    agents = create_test_agents(5)
    tasks = ["Calculate power", "Process power", "Analyze power"]
    
    result = power_swarm(agents, tasks)
    assert result is not None


def test_log_swarm():
    """Test log swarm functionality"""
    agents = create_test_agents(5)
    tasks = ["Calculate log", "Process log", "Analyze log"]
    
    result = log_swarm(agents, tasks)
    assert result is not None


def test_exponential_swarm():
    """Test exponential swarm functionality"""
    agents = create_test_agents(5)
    tasks = ["Calculate exp", "Process exp", "Analyze exp"]
    
    result = exponential_swarm(agents, tasks)
    assert result is not None


def test_geometric_swarm():
    """Test geometric swarm functionality"""
    agents = create_test_agents(5)
    tasks = ["Calculate geo", "Process geo", "Analyze geo"]
    
    result = geometric_swarm(agents, tasks)
    assert result is not None


def test_harmonic_swarm():
    """Test harmonic swarm functionality"""
    agents = create_test_agents(5)
    tasks = ["Calculate harmonic", "Process harmonic", "Analyze harmonic"]
    
    result = harmonic_swarm(agents, tasks)
    assert result is not None

# ============================================================================
# Pattern-Based Swarm Tests
# ============================================================================


def test_staircase_swarm():
    """Test staircase swarm functionality"""
    agents = create_test_agents(6)
    task = "Process according to staircase pattern"
    
    result = staircase_swarm(agents, task)
    assert result is not None


def test_sigmoid_swarm():
    """Test sigmoid swarm functionality"""
    agents = create_test_agents(6)
    task = "Process according to sigmoid pattern"
    
    result = sigmoid_swarm(agents, task)
    assert result is not None


def test_sinusoidal_swarm():
    """Test sinusoidal swarm functionality"""
    agents = create_test_agents(6)
    task = "Process according to sinusoidal pattern"
    
    result = sinusoidal_swarm(agents, task)
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
