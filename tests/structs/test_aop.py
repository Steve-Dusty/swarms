import pytest

from swarms.structs.agent import Agent
from swarms.structs.aop import (
    AOP,
    AOPCluster,
    QueueStatus,
    TaskStatus,
)

# ============================================================================
# Helper Functions
# ============================================================================


def create_test_agent(name: str = "Test-Agent") -> Agent:
    """Create a test agent for AOP testing."""
    return Agent(
        agent_name=name,
        agent_description=f"Test agent {name} for AOP testing",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        temperature=0.5,
        max_tokens=1000,
    )


def create_test_agents(count: int = 3) -> list:
    """Create multiple test agents for batch testing."""
    agents = []
    for i in range(count):
        agent = create_test_agent(f"Test-Agent-{i}")
        agents.append(agent)
    return agents


# ============================================================================
# Enum Tests
# ============================================================================


def test_queue_status_enum():
    """Test QueueStatus enum values."""
    assert QueueStatus.RUNNING.value == "running"
    assert QueueStatus.PAUSED.value == "paused"
    assert QueueStatus.STOPPED.value == "stopped"


def test_task_status_enum():
    """Test TaskStatus enum values."""
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.PROCESSING.value == "processing"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.CANCELLED.value == "cancelled"


# ============================================================================
# AOP Initialization Tests
# ============================================================================


def test_aop_initialization_minimal():
    """Test AOP initialization with minimal parameters."""
    aop = AOP(
        server_name="Test AOP",
        description="Test AOP description",
    )
    
    assert aop.server_name == "Test AOP"
    assert aop.description == "Test AOP description"
    assert aop.port == 8000  # default port
    assert aop.host == "localhost"  # default host
    assert aop.transport == "streamable-http"  # default transport
    assert aop.verbose is False  # default verbose
    assert aop.queue_enabled is True  # default queue_enabled


def test_aop_initialization_with_agents():
    """Test AOP initialization with agents."""
    agents = create_test_agents(2)
    
    aop = AOP(
        server_name="Test AOP with Agents",
        description="Test AOP with multiple agents",
        agents=agents,
        port=8001,
        verbose=False,
    )
    
    assert aop.server_name == "Test AOP with Agents"
    assert aop.description == "Test AOP with multiple agents"
    assert aop.port == 8001
    assert len(aop.agents) == 2


def test_aop_initialization_custom_config():
    """Test AOP initialization with custom configuration."""
    agent = create_test_agent()
    
    aop = AOP(
        server_name="Custom AOP",
        description="Custom AOP configuration",
        agents=[agent],
        port=9000,
        transport="sse",
        verbose=False,
        traceback_enabled=False,
        host="127.0.0.1",
        queue_enabled=True,
        max_workers_per_agent=5,
        max_queue_size_per_agent=200,
        processing_timeout=60,
        retry_delay=2.0,
        persistence=True,
        max_restart_attempts=10,
        restart_delay=10.0,
        network_monitoring=False,
        max_network_retries=5,
        network_retry_delay=15.0,
        network_timeout=20.0,
        log_level="DEBUG",
    )
    
    assert aop.server_name == "Custom AOP"
    assert aop.description == "Custom AOP configuration"
    assert aop.verbose is False
    assert aop.traceback_enabled is False
    assert aop.host == "127.0.0.1"
    assert aop.port == 9000
    assert aop.transport == "sse"
    assert aop.queue_enabled is True
    assert aop.max_workers_per_agent == 5
    assert aop.max_queue_size_per_agent == 200
    assert aop.processing_timeout == 60
    assert aop.retry_delay == 2.0
    assert aop.persistence is True
    assert aop.max_restart_attempts == 10
    assert aop.restart_delay == 10.0
    assert aop.network_monitoring is False
    assert aop.max_network_retries == 5
    assert aop.network_retry_delay == 15.0
    assert aop.network_timeout == 20.0
    assert aop.log_level == "DEBUG"


# ============================================================================
# Agent Management Tests
# ============================================================================


def test_aop_add_agent():
    """Test adding an agent to AOP."""
    aop = AOP(
        server_name="Agent Management Test",
        description="Test agent management",
    )
    
    agent = create_test_agent("NewAgent")
    aop.add_agent(agent)
    
    assert "NewAgent" in aop.agents
    assert aop.agents["NewAgent"] == agent


def test_aop_add_agents_batch():
    """Test adding multiple agents to AOP."""
    aop = AOP(
        server_name="Batch Agent Test",
        description="Test batch agent addition",
    )
    
    agents = create_test_agents(3)
    aop.add_agents_batch(agents)
    
    assert len(aop.agents) == 3
    for agent in agents:
        assert agent.agent_name in aop.agents


def test_aop_remove_agent():
    """Test removing an agent from AOP."""
    agent = create_test_agent("RemoveAgent")
    
    aop = AOP(
        server_name="Agent Removal Test",
        description="Test agent removal",
        agents=[agent],
    )
    
    # Verify agent was added
    assert "RemoveAgent" in aop.agents
    
    # Remove the agent
    aop.remove_agent("RemoveAgent")
    
    # Verify agent was removed
    assert "RemoveAgent" not in aop.agents


def test_aop_agents_dictionary():
    """Test that agents are stored as dictionary in AOP."""
    agents = create_test_agents(2)
    
    aop = AOP(
        server_name="Agents Dict Test",
        description="Test agent dictionary storage",
        agents=agents,
    )
    
    # AOP stores agents as a dictionary mapping tool names to agents
    assert isinstance(aop.agents, dict)
    assert len(aop.agents) == 2
    # Check that agent names are in the keys
    agent_names = [agent.agent_name for agent in agents]
    for name in agent_names:
        assert name in aop.agents


def test_aop_get_agent_info():
    """Test getting agent information."""
    agent = create_test_agent("InfoAgent")
    
    aop = AOP(
        server_name="Agent Info Test",
        description="Test agent info retrieval",
        agents=[agent],
    )
    
    info = aop.get_agent_info("InfoAgent")
    assert info is not None
    assert "InfoAgent" in str(info)


# ============================================================================
# Server Management Tests
# ============================================================================


def test_aop_get_server_info():
    """Test getting server information."""
    aop = AOP(
        server_name="Server Info Test",
        description="Test server info retrieval",
    )
    
    info = aop.get_server_info()
    assert isinstance(info, dict)
    assert "server_name" in info
    assert info["server_name"] == "Server Info Test"


# ============================================================================
# Queue Management Tests
# ============================================================================


def test_aop_get_queue_stats():
    """Test getting queue statistics."""
    agent = create_test_agent("QueueAgent")
    
    aop = AOP(
        server_name="Queue Stats Test",
        description="Test queue statistics",
        agents=[agent],
        queue_enabled=True,
    )
    
    stats = aop.get_queue_stats("QueueAgent")
    assert stats is not None


def test_aop_pause_resume_queue():
    """Test pausing and resuming agent queue."""
    agent = create_test_agent("PauseAgent")
    
    aop = AOP(
        server_name="Pause Resume Test",
        description="Test queue pause/resume",
        agents=[agent],
        queue_enabled=True,
    )
    
    # Test pause
    aop.pause_agent_queue("PauseAgent")
    
    # Test resume
    aop.resume_agent_queue("PauseAgent")


def test_aop_clear_queue():
    """Test clearing agent queue."""
    agent = create_test_agent("ClearAgent")
    
    aop = AOP(
        server_name="Clear Queue Test",
        description="Test queue clearing",
        agents=[agent],
        queue_enabled=True,
    )
    
    cleared_count = aop.clear_agent_queue("ClearAgent")
    assert isinstance(cleared_count, int)
    assert cleared_count >= 0


# ============================================================================
# AOPCluster Tests
# ============================================================================


def test_aop_cluster_initialization():
    """Test AOPCluster initialization."""
    urls = ["http://localhost:8000/mcp", "http://localhost:8001/mcp"]
    
    cluster = AOPCluster(
        urls=urls,
        transport="streamable-http",
    )
    
    assert cluster.urls == urls
    assert cluster.transport == "streamable-http"


def test_aop_cluster_get_tools():
    """Test getting tools from AOPCluster."""
    urls = ["http://localhost:8000/mcp"]
    
    cluster = AOPCluster(
        urls=urls,
        transport="streamable-http",
    )
    
    # This will fail if no servers are running, but we test the method exists
    try:
        tools = cluster.get_tools(output_type="dict")
        assert isinstance(tools, list)
    except Exception as e:
        # Expected if no servers are running - check for connection-related errors
        error_msg = str(e).lower()
        assert any(keyword in error_msg for keyword in [
            "connection", "network", "connect", "failed to fetch", "mcp"
        ])


def test_aop_cluster_find_tool_by_server_name():
    """Test finding tool by server name in AOPCluster."""
    urls = ["http://localhost:8000/mcp"]
    
    cluster = AOPCluster(
        urls=urls,
        transport="streamable-http",
    )
    
    # This will fail if no servers are running, but we test the method exists
    try:
        tool = cluster.find_tool_by_server_name("Test-Agent")
        # If successful, should return dict or None
        assert tool is None or isinstance(tool, dict)
    except Exception as e:
        # Expected if no servers are running - check for connection-related errors
        error_msg = str(e).lower()
        assert any(keyword in error_msg for keyword in [
            "connection", "network", "connect", "failed to fetch", "mcp"
        ])


# ============================================================================
# Configuration Validation Tests
# ============================================================================


def test_aop_port_configuration():
    """Test AOP with different port configurations."""
    # Test with valid integer port
    aop1 = AOP(
        server_name="Port Test 1",
        description="Test valid port",
        port=8080,
    )
    assert aop1.port == 8080
    
    # Test with default port
    aop2 = AOP(
        server_name="Port Test 2",
        description="Test default port",
    )
    assert aop2.port == 8000  # default port


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_aop_server_name_variations():
    """Test AOP with different server name configurations."""
    # Test with empty string (AOP accepts this)
    aop1 = AOP(
        server_name="",
        description="Test empty name",
    )
    assert aop1.server_name == ""
    
    # Test with None (AOP converts to string)
    aop2 = AOP(
        server_name=None,
        description="Test None name",
    )
    assert aop2.server_name is None or aop2.server_name == "None"


# ============================================================================
# Integration Tests
# ============================================================================


def test_aop_full_workflow():
    """Test complete AOP workflow."""
    agents = create_test_agents(2)
    
    # Initialize AOP
    aop = AOP(
        server_name="Integration Test AOP",
        description="Complete workflow test",
        agents=agents,
        port=8002,
        verbose=False,
        queue_enabled=True,
    )
    
    # Add additional agent
    extra_agent = create_test_agent("ExtraAgent")
    aop.add_agent(extra_agent)
    
    # Verify agent was added
    assert "ExtraAgent" in aop.agents
    
    # Test server info
    info = aop.get_server_info()
    assert isinstance(info, dict)
    assert info["server_name"] == "Integration Test AOP"
    
    # Test queue operations
    stats = aop.get_queue_stats("ExtraAgent")
    assert stats is not None
    
    # Remove agent
    aop.remove_agent("ExtraAgent")
    assert "ExtraAgent" not in aop.agents


def test_aop_cluster_workflow():
    """Test complete AOPCluster workflow."""
    urls = [
        "http://localhost:8000/mcp",
        "http://localhost:8001/mcp",
    ]
    
    # Create cluster
    cluster = AOPCluster(
        urls=urls,
        transport="streamable-http",
    )
    
    assert cluster.urls == urls
    assert cluster.transport == "streamable-http"
    
    # Test methods exist and have proper signatures
    assert hasattr(cluster, "get_tools")
    assert hasattr(cluster, "find_tool_by_server_name")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])