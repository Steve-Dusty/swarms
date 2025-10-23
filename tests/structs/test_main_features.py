import pytest
from typing import List, Callable

from swarms.structs import (
    Agent,
    AgentRearrange,
    ConcurrentWorkflow,
    GroupChat,
    InteractiveGroupChat,
    MajorityVoting,
    MixtureOfAgents,
    MultiAgentRouter,
    RoundRobinSwarm,
    SequentialWorkflow,
    SpreadSheetSwarm,
    SwarmRouter,
)
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.tree_swarm import ForestSwarm, Tree, TreeAgent

# ============================================================================
# Helper Functions
# ============================================================================


def create_test_agent(
    name: str,
    system_prompt: str = None,
    model_name: str = "gpt-4o-mini",
    tools: List[Callable] = None,
    **kwargs,
) -> Agent:
    """Create a properly configured test agent."""
    return Agent(
        agent_name=name,
        system_prompt=system_prompt or f"You are {name}, a helpful AI assistant.",
        model_name=model_name,
        max_loops=1,
        max_tokens=200,
        verbose=False,
        print_on=False,
        tools=tools or [],
        **kwargs,
    )


def create_test_agents(count: int, prefix: str = "Agent") -> List[Agent]:
    """Create multiple test agents."""
    return [
        create_test_agent(f"{prefix}-{i+1}") for i in range(count)
    ]

# ============================================================================
# Basic Agent Tests
# ============================================================================


def test_basic_agent_functionality():
    """Test basic agent creation and execution."""
    agent = create_test_agent("TestAgent")
    
    assert agent.agent_name == "TestAgent"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.max_loops == 1
    
    result = agent.run("What is 2+2?")
    assert result is not None
    assert isinstance(result, str)


def test_agent_with_custom_prompt():
    """Test agent with custom system prompt."""
    custom_prompt = "You are a math expert. Always show your work."
    agent = create_test_agent(
        "MathExpert",
        system_prompt=custom_prompt,
    )
    
    assert agent.system_prompt == custom_prompt
    
    result = agent.run("Calculate 15 * 7")
    assert result is not None
    assert isinstance(result, str)


def test_tool_execution_with_agent():
    """Test agent with tool execution capabilities."""
    def simple_calculator(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    agent = create_test_agent(
        "ToolAgent",
        tools=[simple_calculator],
    )
    
    result = agent.run("Use the calculator to add 5 and 3")
    assert result is not None


def test_multimodal_execution():
    """Test agent with multimodal capabilities."""
    agent = create_test_agent("MultimodalAgent")
    
    # Test with text only
    result = agent.run("Describe the concept of AI")
    assert result is not None
    
    # Test with None image (no actual image processing)
    result = agent.run("Describe this image", img=None)
    assert result is not None

# ============================================================================
# Workflow Tests
# ============================================================================


def test_sequential_workflow():
    """Test SequentialWorkflow with multiple agents."""
    agents = create_test_agents(3, "Sequential")
    
    workflow = SequentialWorkflow(
        name="Test-Sequential-Workflow",
        description="Test sequential workflow execution",
        agents=agents,
        max_loops=1,
    )
    
    result = workflow.run("Process this task sequentially")
    assert result is not None


def test_concurrent_workflow():
    """Test ConcurrentWorkflow with multiple agents."""
    agents = create_test_agents(3, "Concurrent")
    
    workflow = ConcurrentWorkflow(
        name="Test-Concurrent-Workflow", 
        description="Test concurrent workflow execution",
        agents=agents,
        max_loops=1,
    )
    
    result = workflow.run("Process this task concurrently")
    assert result is not None
    assert isinstance(result, list)


def test_agent_rearrange():
    """Test AgentRearrange functionality."""
    agents = create_test_agents(2, "Rearrange")
    
    rearrange = AgentRearrange(
        name="Test-Agent-Rearrange",
        description="Test agent rearrangement",
        agents=agents,
        flow=f"{agents[0].agent_name} -> {agents[1].agent_name}",
        max_loops=1,
    )
    
    result = rearrange.run("Test rearrangement flow")
    assert result is not None

# ============================================================================
# Swarm Structure Tests
# ============================================================================


def test_mixture_of_agents():
    """Test MixtureOfAgents functionality."""
    agents = create_test_agents(3, "MOA")
    aggregator = create_test_agent("Aggregator")
    
    moa = MixtureOfAgents(
        name="Test-MOA",
        description="Test mixture of agents",
        agents=agents,
        aggregator_agent=aggregator,
        layers=2,
        max_loops=1,
    )
    
    result = moa.run("Test mixture of agents")
    assert result is not None


def test_spreadsheet_swarm():
    """Test SpreadSheetSwarm functionality."""
    agent = create_test_agent("SpreadsheetAgent")
    
    swarm = SpreadSheetSwarm(
        name="Test-Spreadsheet-Swarm",
        description="Test spreadsheet swarm",
        agents=[agent],
        max_loops=1,
        autosave=False,
    )
    
    result = swarm.run("Test spreadsheet operations")
    assert result is not None


def test_hierarchical_swarm():
    """Test HierarchicalSwarm functionality."""
    agents = create_test_agents(4, "Hierarchical")
    
    swarm = HierarchicalSwarm(
        name="Test-Hierarchical-Swarm",
        description="Test hierarchical swarm structure",
        agents=agents,
        max_loops=1,
        verbose=False,
    )
    
    result = swarm.run("Test hierarchical processing")
    assert result is not None


def test_majority_voting():
    """Test MajorityVoting functionality."""
    agents = create_test_agents(3, "Voting")
    
    voting = MajorityVoting(
        name="Test-Majority-Voting",
        description="Test majority voting system",
        agents=agents,
        max_loops=1,
        verbose=False,
    )
    
    result = voting.run("What is the capital of France?")
    assert result is not None


def test_round_robin_swarm():
    """Test RoundRobinSwarm functionality."""
    agents = create_test_agents(3, "RoundRobin")
    
    swarm = RoundRobinSwarm(
        name="Test-Round-Robin-Swarm",
        description="Test round robin swarm",
        agents=agents,
        max_loops=1,
        verbose=False,
    )
    
    result = swarm.run("Test round robin processing")
    assert result is not None

# ============================================================================
# Router Tests
# ============================================================================


def test_swarm_router():
    """Test SwarmRouter functionality."""
    agents = create_test_agents(2, "Router")
    
    router = SwarmRouter(
        name="Test-Swarm-Router",
        description="Test swarm router",
        agents=agents,
        swarm_type="SequentialWorkflow",
        max_loops=1,
        verbose=False,
    )
    
    result = router.run("Test routing functionality")
    assert result is not None


def test_multi_agent_router():
    """Test MultiAgentRouter functionality."""
    agents = create_test_agents(3, "MultiRouter")
    
    router = MultiAgentRouter(
        name="Test-Multi-Agent-Router",
        description="Test multi-agent router",
        agents=agents,
        max_loops=1,
    )
    
    result = router.run("Test multi-agent routing")
    assert result is not None

# ============================================================================
# Communication Tests
# ============================================================================


def test_groupchat():
    """Test GroupChat functionality."""
    agents = create_test_agents(3, "GroupChat")
    
    groupchat = GroupChat(
        name="Test-Group-Chat",
        description="Test group chat functionality",
        agents=agents,
        max_loops=1,
        selector_agent=agents[0],
    )
    
    result = groupchat.run("Test group discussion")
    assert result is not None


def test_interactive_groupchat():
    """Test InteractiveGroupChat functionality."""
    agents = create_test_agents(2, "Interactive")
    
    interactive_chat = InteractiveGroupChat(
        name="Test-Interactive-Chat",
        description="Test interactive group chat",
        agents=agents,
        max_loops=1,
    )
    
    # Test basic initialization
    assert interactive_chat.name == "Test-Interactive-Chat"
    assert len(interactive_chat.agents) == 2

# ============================================================================
# Tree Structure Tests
# ============================================================================


def test_forest_swarm():
    """Test ForestSwarm functionality."""
    # Create tree agents
    tree_agents = []
    for i in range(3):
        agent = create_test_agent(f"TreeAgent-{i+1}")
        tree_agent = TreeAgent(
            system_prompt=f"You are tree agent {i+1}",
            agent=agent,
        )
        tree_agents.append(tree_agent)
    
    # Create trees
    trees = []
    for i in range(2):
        tree = Tree(
            tree_name=f"Tree-{i+1}",
            agents=tree_agents,
        )
        trees.append(tree)
    
    # Create forest
    forest = ForestSwarm(
        trees=trees,
        verbose=False,
    )
    
    result = forest.run("Test forest processing")
    assert result is not None

# ============================================================================
# Advanced Feature Tests
# ============================================================================


def test_streaming_mode():
    """Test agent streaming capabilities."""
    agent = create_test_agent("StreamingAgent")
    
    # Test basic streaming (agent handles streaming internally)
    result = agent.run("Generate a short story")
    assert result is not None
    assert isinstance(result, str)


def test_agent_memory_persistence():
    """Test agent memory and conversation persistence."""
    agent = create_test_agent("MemoryAgent")
    
    # First interaction
    result1 = agent.run("My name is Alice")
    assert result1 is not None
    
    # Second interaction (memory may or may not persist depending on agent config)
    result2 = agent.run("What is my name?")
    assert result2 is not None


def test_error_handling():
    """Test error handling across different components."""
    agent = create_test_agent("ErrorTestAgent")
    
    # Test with empty task
    with pytest.raises((ValueError, TypeError)):
        agent.run("")
    
    # Test with None task
    with pytest.raises((ValueError, TypeError)):
        agent.run(None)
    
    # Test invalid workflow configuration
    with pytest.raises((ValueError, TypeError)):
        SequentialWorkflow(agents=None)

# ============================================================================
# Integration Tests
# ============================================================================


def test_complex_workflow_integration():
    """Test complex multi-component workflow integration."""
    # Create specialized agents
    research_agent = create_test_agent(
        "Research-Agent",
        system_prompt="You specialize in research and data gathering.",
    )
    
    analysis_agent = create_test_agent(
        "Analysis-Agent", 
        system_prompt="You specialize in data analysis and insights.",
    )
    
    summary_agent = create_test_agent(
        "Summary-Agent",
        system_prompt="You specialize in creating executive summaries.",
    )
    
    # Create sequential workflow
    workflow = SequentialWorkflow(
        name="Complex-Integration-Workflow",
        description="Multi-stage research and analysis workflow",
        agents=[research_agent, analysis_agent, summary_agent],
        max_loops=1,
    )
    
    # Execute complex task
    result = workflow.run(
        "Research the benefits of renewable energy, analyze the data, and provide an executive summary"
    )
    
    assert result is not None
    assert isinstance(result, str)
    
    # Test with mixture of agents for comparison
    moa = MixtureOfAgents(
        name="Integration-MOA",
        description="Mixture of agents for comparison",
        agents=[research_agent, analysis_agent],
        aggregator_agent=summary_agent,
        layers=2,
        max_loops=1,
    )
    
    moa_result = moa.run("Compare renewable vs fossil fuel energy sources")
    assert moa_result is not None


def test_multi_swarm_coordination():
    """Test coordination between multiple swarm types."""
    agents = create_test_agents(4, "Coordination")
    
    # Create different swarm types
    sequential = SequentialWorkflow(
        name="Sequential-Coordinator",
        agents=agents[:2],
        max_loops=1,
    )
    
    concurrent = ConcurrentWorkflow(
        name="Concurrent-Coordinator", 
        agents=agents[2:],
        max_loops=1,
    )
    
    # Test both swarms on related tasks
    seq_result = sequential.run("Analyze market trends step by step")
    conc_result = concurrent.run("Analyze market trends from multiple perspectives")
    
    assert seq_result is not None
    assert conc_result is not None
    assert isinstance(conc_result, list)


def test_agent_specialization():
    """Test agents with different specializations working together."""
    # Create specialized agents
    math_agent = create_test_agent(
        "Math-Specialist",
        system_prompt="You are a mathematics expert. Solve problems step by step.",
    )
    
    writing_agent = create_test_agent(
        "Writing-Specialist", 
        system_prompt="You are a writing expert. Create clear, engaging content.",
    )
    
    code_agent = create_test_agent(
        "Code-Specialist",
        system_prompt="You are a programming expert. Write clean, efficient code.",
    )
    
    # Test individual specializations
    math_result = math_agent.run("Calculate the area of a circle with radius 5")
    writing_result = writing_agent.run("Write a brief explanation of photosynthesis")
    code_result = code_agent.run("Write a Python function to calculate fibonacci numbers")
    
    assert math_result is not None
    assert writing_result is not None
    assert code_result is not None
    
    # Test collaboration
    workflow = SequentialWorkflow(
        name="Specialist-Collaboration",
        agents=[math_agent, writing_agent, code_agent],
        max_loops=1,
    )
    
    collab_result = workflow.run(
        "Create a tutorial that explains the math behind fibonacci numbers, "
        "writes it clearly, and provides code examples"
    )
    
    assert collab_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])