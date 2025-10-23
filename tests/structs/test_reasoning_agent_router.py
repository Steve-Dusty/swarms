import pytest

from swarms.agents.reasoning_agents import (
    ReasoningAgentInitializationError,
    ReasoningAgentRouter,
)


def test_router_initialization_default():
    """Test default ReasoningAgentRouter initialization"""
    router = ReasoningAgentRouter()

    assert router.agent_name == "reasoning_agent"
    assert router.swarm_type == "reasoning-duo"
    assert router.model_name == "gpt-4o-mini"


def test_router_initialization_custom_parameters():
    """Test ReasoningAgentRouter with custom parameters"""
    custom_router = ReasoningAgentRouter(
        agent_name="test_agent",
        description="Test agent for unit testing",
        model_name="gpt-4",
        system_prompt="You are a test agent.",
        max_loops=5,
        swarm_type="self-consistency",
        num_samples=3,
        output_type="dict-all-except-first",
        num_knowledge_items=10,
        memory_capacity=20,
        eval=True,
        random_models_on=True,
        majority_voting_prompt="Custom voting prompt",
        reasoning_model_name="claude-3-5-sonnet-20240620"
    )

    assert custom_router.agent_name == "test_agent"
    assert custom_router.swarm_type == "self-consistency"
    assert custom_router.max_loops == 5
    assert custom_router.num_samples == 3


def test_router_initialization_all_agent_types():
    """Test initialization for all supported agent types"""
    agent_types = [
        "reasoning-duo",
        "reasoning-agent",
        "self-consistency",
        "consistency-agent",
        "ire",
        "ire-agent",
        "ReflexionAgent",
        "GKPAgent",
        "AgentJudge"
    ]

    for agent_type in agent_types:
        router = ReasoningAgentRouter(swarm_type=agent_type)
        assert router.swarm_type == agent_type


def test_reliability_check_zero_max_loops():
    """Test that zero max_loops raises error"""
    with pytest.raises(ReasoningAgentInitializationError, match="Max loops must be greater than 0"):
        ReasoningAgentRouter(max_loops=0)


def test_reliability_check_empty_model_name():
    """Test that empty model_name raises error"""
    with pytest.raises(ReasoningAgentInitializationError, match="Model name must be provided"):
        ReasoningAgentRouter(model_name="")


def test_reliability_check_none_model_name():
    """Test that None model_name raises error"""
    with pytest.raises(ReasoningAgentInitializationError, match="Model name must be provided"):
        ReasoningAgentRouter(model_name=None)


def test_reliability_check_empty_swarm_type():
    """Test that empty swarm_type raises error"""
    with pytest.raises(ReasoningAgentInitializationError, match="Swarm type must be provided"):
        ReasoningAgentRouter(swarm_type="")


def test_reliability_check_none_swarm_type():
    """Test that None swarm_type raises error"""
    with pytest.raises(ReasoningAgentInitializationError, match="Swarm type must be provided"):
        ReasoningAgentRouter(swarm_type=None)


def test_create_reasoning_duo():
    """Test _create_reasoning_duo factory method"""
    router = ReasoningAgentRouter(
        swarm_type="reasoning-duo",
        agent_name="test_agent",
        model_name="gpt-4o-mini",
        max_loops=2
    )
    agent = router._create_reasoning_duo()
    assert agent is not None


def test_create_consistency_agent():
    """Test _create_consistency_agent factory method"""
    router = ReasoningAgentRouter(
        swarm_type="self-consistency",
        agent_name="test_agent",
        model_name="gpt-4o-mini",
        num_samples=3
    )
    agent = router._create_consistency_agent()
    assert agent is not None


def test_create_ire_agent():
    """Test _create_ire_agent factory method"""
    router = ReasoningAgentRouter(
        swarm_type="ire",
        agent_name="test_agent",
        model_name="gpt-4o-mini",
        max_loops=2
    )
    agent = router._create_ire_agent()
    assert agent is not None


def test_create_agent_judge():
    """Test _create_agent_judge factory method"""
    router = ReasoningAgentRouter(
        swarm_type="AgentJudge",
        agent_name="test_agent",
        model_name="gpt-4o-mini",
        max_loops=2
    )
    agent = router._create_agent_judge()
    assert agent is not None


def test_create_reflexion_agent():
    """Test _create_reflexion_agent factory method"""
    router = ReasoningAgentRouter(
        swarm_type="ReflexionAgent",
        agent_name="test_agent",
        model_name="gpt-4o-mini",
        max_loops=2
    )
    agent = router._create_reflexion_agent()
    assert agent is not None


def test_create_gkp_agent():
    """Test _create_gkp_agent factory method"""
    router = ReasoningAgentRouter(
        swarm_type="GKPAgent",
        agent_name="test_agent",
        model_name="gpt-4o-mini"
    )
    agent = router._create_gkp_agent()
    assert agent is not None


def test_select_swarm_valid_types():
    """Test select_swarm for all valid agent types"""
    agent_types = [
        "reasoning-duo",
        "reasoning-agent",
        "self-consistency",
        "consistency-agent",
        "ire",
        "ire-agent",
        "ReflexionAgent",
        "GKPAgent",
        "AgentJudge"
    ]

    for agent_type in agent_types:
        router = ReasoningAgentRouter(swarm_type=agent_type)
        swarm = router.select_swarm()
        assert swarm is not None


def test_select_swarm_invalid_type():
    """Test that invalid swarm type raises error"""
    router = ReasoningAgentRouter(swarm_type="invalid_type")

    with pytest.raises(ReasoningAgentInitializationError, match="Invalid swarm type"):
        router.select_swarm()


def test_run_method_exists():
    """Test that run method exists and is callable"""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")

    assert hasattr(router, 'run')
    assert callable(router.run)


def test_run_method_signature():
    """Test run method has correct signature"""
    import inspect

    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    sig = inspect.signature(router.run)

    assert 'task' in sig.parameters


def test_batched_run_method_exists():
    """Test that batched_run method exists"""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")

    assert hasattr(router, 'batched_run')
    assert callable(router.batched_run)


def test_batched_run_method_signature():
    """Test batched_run method has correct signature"""
    import inspect

    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    sig = inspect.signature(router.batched_run)

    assert 'tasks' in sig.parameters


def test_output_types_configuration():
    """Test different output type configurations"""
    output_types = [
        "dict-all-except-first",
        "dict",
        "string",
        "list"
    ]

    for output_type in output_types:
        router = ReasoningAgentRouter(
            swarm_type="reasoning-duo",
            output_type=output_type
        )
        assert router.output_type == output_type


def test_num_samples_configuration():
    """Test num_samples configuration"""
    router = ReasoningAgentRouter(
        swarm_type="self-consistency",
        num_samples=5
    )
    assert router.num_samples == 5


def test_max_loops_configuration():
    """Test max_loops configuration"""
    router = ReasoningAgentRouter(
        swarm_type="reasoning-duo",
        max_loops=10
    )
    assert router.max_loops == 10


def test_memory_capacity_configuration():
    """Test memory_capacity configuration"""
    router = ReasoningAgentRouter(
        swarm_type="ReflexionAgent",
        memory_capacity=50
    )
    assert router.memory_capacity == 50


def test_num_knowledge_items_configuration():
    """Test num_knowledge_items configuration"""
    router = ReasoningAgentRouter(
        swarm_type="GKPAgent",
        num_knowledge_items=15
    )
    assert router.num_knowledge_items == 15


def test_router_all_agent_types_basic():
    """Test basic initialization for all agent types"""
    agent_types = [
        "reasoning-duo",
        "self-consistency",
        "ire",
        "ReflexionAgent",
        "GKPAgent",
        "AgentJudge"
    ]

    for agent_type in agent_types:
        router = ReasoningAgentRouter(swarm_type=agent_type, max_loops=1)
        assert router is not None
        assert router.swarm_type == agent_type
        assert hasattr(router, 'run')


def test_reasoning_duo_specific_config():
    """Test reasoning-duo with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="reasoning-duo",
        agent_name="duo_test",
        max_loops=2,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "reasoning-duo"
    assert router.agent_name == "duo_test"
    assert router.max_loops == 2


def test_self_consistency_specific_config():
    """Test self-consistency with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="self-consistency",
        num_samples=5,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "self-consistency"
    assert router.num_samples == 5


def test_ire_agent_specific_config():
    """Test IRE agent with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="ire",
        max_loops=3,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "ire"
    assert router.max_loops == 3


def test_reflexion_agent_specific_config():
    """Test Reflexion agent with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="ReflexionAgent",
        memory_capacity=100,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "ReflexionAgent"
    assert router.memory_capacity == 100


def test_gkp_agent_specific_config():
    """Test GKP agent with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="GKPAgent",
        num_knowledge_items=20,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "GKPAgent"
    assert router.num_knowledge_items == 20


def test_agent_judge_specific_config():
    """Test Agent Judge with specific configuration"""
    router = ReasoningAgentRouter(
        swarm_type="AgentJudge",
        max_loops=2,
        model_name="gpt-4o-mini"
    )

    assert router.swarm_type == "AgentJudge"
    assert router.max_loops == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
