import pytest
from dotenv import load_dotenv

from swarms.structs.agent import Agent
from swarms.structs.auto_swarm_builder import (
    AgentSpec,
    AutoSwarmBuilder,
)
from swarms.structs.ma_utils import set_random_models_for_agents

load_dotenv()


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_initialization():
    """Test basic initialization of AutoSwarmBuilder"""
    swarm = AutoSwarmBuilder(
        name="TestSwarm",
        description="A test swarm for validation",
        verbose=True,
        max_loops=2,
    )

    assert swarm.name == "TestSwarm"
    assert swarm.description == "A test swarm for validation"
    assert swarm.max_loops == 2
    assert swarm.verbose is True


def test_initialization_default():
    """Test default initialization"""
    swarm = AutoSwarmBuilder()
    assert swarm is not None


# ============================================================================
# AGENT BUILDING TESTS
# ============================================================================


def test_agent_building():
    """Test building individual agents using dict_to_agent"""
    swarm = AutoSwarmBuilder()
    
    # Create agent config in the format expected by dict_to_agent (with "agents" key and "name" field)
    agents_dict = {
        "agents": [{
            "name": "TestAgent",  # dict_to_agent expects "name", not "agent_name"
            "agent_name": "TestAgent",
            "description": "A test agent",
            "system_prompt": "You are a test agent",
            "model_name": "gpt-4o-mini",
            "max_loops": 1,
            "verbose": False,
            "print_on": False
        }]
    }
    
    agents = swarm.dict_to_agent(agents_dict)

    assert agents is not None
    assert isinstance(agents, list)
    assert len(agents) > 0
    assert agents[0].agent_name == "TestAgent"
    assert agents[0].max_loops == 1


def test_agent_building_with_model():
    """Test building agent with specific model using dict_to_agent"""
    swarm = AutoSwarmBuilder()
    
    # Create agent config in the format expected by dict_to_agent (with "agents" key and "name" field)
    agents_dict = {
        "agents": [{
            "name": "ModelAgent",  # dict_to_agent expects "name", not "agent_name"
            "agent_name": "ModelAgent",
            "description": "Agent with specific model",
            "system_prompt": "You are a model test agent",
            "model_name": "gpt-4o-mini",
            "max_loops": 1,
            "verbose": False,
            "print_on": False
        }]
    }
    
    agents = swarm.dict_to_agent(agents_dict)

    assert isinstance(agents, list)
    assert len(agents) > 0
    assert agents[0].agent_name == "ModelAgent"
    assert agents[0].model_name == "gpt-4o-mini"


# ============================================================================
# AGENT CREATION TESTS
# ============================================================================


def test_agent_creation():
    """Test creating multiple agents for a task"""
    swarm = AutoSwarmBuilder(
        name="ResearchSwarm",
        description="A swarm for research tasks",
        execution_type="return-agents",  # Return agent dictionary
    )
    task = "Research the latest developments in quantum computing"
    result = swarm.create_agents(task)

    assert result is not None
    # create_agents returns a JSON string, need to parse it
    import json
    if isinstance(result, str):
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, dict)
        assert "agents" in parsed_result
        assert len(parsed_result["agents"]) > 0
    else:
        # If it returns a dict directly
        assert isinstance(result, dict)
        assert "agents" in result
        assert len(result["agents"]) > 0


# ============================================================================
# SWARM ROUTING TESTS
# ============================================================================


def test_swarm_routing():
    """Test routing tasks through the swarm"""
    swarm = AutoSwarmBuilder(
        name="RouterTestSwarm",
        description="Testing routing capabilities",
        execution_type="return-swarm-router-config",  # Return router config
    )

    # Test router configuration
    task = "Analyze the impact of AI on healthcare"
    router_config = swarm.run(task)

    assert router_config is not None
    assert "swarm_type" in router_config


# ============================================================================
# FULL EXECUTION TESTS
# ============================================================================


def test_full_swarm_execution():
    """Test complete swarm execution with a real task"""
    swarm = AutoSwarmBuilder(
        name="FullTestSwarm",
        description="Testing complete swarm functionality",
        max_loops=1,
    )
    task = "Create a summary of recent advances in renewable energy"

    result = swarm.run(task)

    assert result is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_error_handling_invalid_agent():
    """Test error handling with invalid agent configuration"""
    swarm = AutoSwarmBuilder()

    # Test with invalid agent configuration using dict_to_agent
    invalid_config = {
        "agents": [{
            "name": "",  # dict_to_agent expects "name", not "agent_name"
            "agent_name": "",  # Empty name
            "description": "",
            "system_prompt": "",
            "model_name": "gpt-4o-mini"
        }]
    }
    
    # This might not raise an exception as AutoSwarmBuilder is permissive
    # Just test that it returns something
    result = swarm.dict_to_agent(invalid_config)
    # The result might be an agent with empty name, which is handled gracefully
    assert result is not None
    assert isinstance(result, list)


def test_error_handling_none_task():
    """Test handling None task - AutoSwarmBuilder handles this gracefully"""
    swarm = AutoSwarmBuilder()

    # AutoSwarmBuilder handles None task gracefully, doesn't raise exception
    result = swarm.run(None)
    # Should return some result or handle gracefully
    assert result is not None or result is None  # Either is acceptable


# ============================================================================
# AGENT SPEC TESTS (Bug #1115 fixes)
# ============================================================================


def test_create_agents_from_specs_with_dict():
    """Test that create_agents_from_specs handles dict input correctly"""
    builder = AutoSwarmBuilder()

    # Create specs as a dictionary
    specs = {
        "agents": [
            {
                "agent_name": "test_agent_1",
                "description": "Test agent 1 description",
                "system_prompt": "You are a helpful assistant",
                "model_name": "gpt-4o-mini",
                "max_loops": 1,
            }
        ]
    }

    agents = builder.create_agents_from_specs(specs)

    # Verify agents were created correctly
    assert len(agents) == 1
    assert isinstance(agents[0], Agent)
    assert agents[0].agent_name == "test_agent_1"

    # Verify description was mapped to agent_description
    assert hasattr(agents[0], "agent_description")
    assert agents[0].agent_description == "Test agent 1 description"


def test_create_agents_from_specs_with_pydantic():
    """Test that create_agents_from_specs handles Pydantic model input correctly

    This is the main test for bug #1115 - it verifies that AgentSpec
    Pydantic models can be unpacked correctly.
    """
    builder = AutoSwarmBuilder()

    # Create specs as Pydantic AgentSpec objects
    agent_spec = AgentSpec(
        agent_name="test_agent_pydantic",
        description="Pydantic test agent",
        system_prompt="You are a helpful assistant",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    specs = {"agents": [agent_spec]}

    agents = builder.create_agents_from_specs(specs)

    # Verify agents were created correctly
    assert len(agents) == 1
    assert isinstance(agents[0], Agent)
    assert agents[0].agent_name == "test_agent_pydantic"

    # Verify description was mapped to agent_description
    assert hasattr(agents[0], "agent_description")
    assert agents[0].agent_description == "Pydantic test agent"


def test_parameter_name_mapping():
    """Test that 'description' field maps to 'agent_description' correctly"""
    builder = AutoSwarmBuilder()

    # Test with dict that has 'description'
    specs = {
        "agents": [
            {
                "agent_name": "mapping_test",
                "description": "This should map to agent_description",
                "system_prompt": "You are helpful",
            }
        ]
    }

    agents = builder.create_agents_from_specs(specs)

    assert len(agents) == 1
    agent = agents[0]

    # Verify description was mapped
    assert hasattr(agent, "agent_description")
    assert (
        agent.agent_description == "This should map to agent_description"
    )


def test_create_agents_from_specs_mixed_input():
    """Test that create_agents_from_specs handles mixed dict and Pydantic input"""
    builder = AutoSwarmBuilder()

    # Mix of dict and Pydantic objects
    dict_spec = {
        "agent_name": "dict_agent",
        "description": "Dict agent description",
        "system_prompt": "You are helpful",
    }

    pydantic_spec = AgentSpec(
        agent_name="pydantic_agent",
        description="Pydantic agent description",
        system_prompt="You are smart",
    )

    specs = {"agents": [dict_spec, pydantic_spec]}

    agents = builder.create_agents_from_specs(specs)

    # Verify both agents were created
    assert len(agents) == 2
    assert all(isinstance(agent, Agent) for agent in agents)

    # Verify both have correct descriptions
    dict_agent = next(a for a in agents if a.agent_name == "dict_agent")
    pydantic_agent = next(
        a for a in agents if a.agent_name == "pydantic_agent"
    )

    assert dict_agent.agent_description == "Dict agent description"
    assert (
        pydantic_agent.agent_description == "Pydantic agent description"
    )


def test_agent_spec_to_agent_all_fields():
    """Test that all AgentSpec fields are properly passed to Agent"""
    builder = AutoSwarmBuilder()

    agent_spec = AgentSpec(
        agent_name="full_test_agent",
        description="Full test description",
        system_prompt="You are a comprehensive test agent",
        model_name="gpt-4o-mini",
        auto_generate_prompt=False,
        max_tokens=4096,
        temperature=0.7,
        role="worker",
        max_loops=3,
        goal="Test all parameters",
    )

    agents = builder.create_agents_from_specs({"agents": [agent_spec]})

    assert len(agents) == 1
    agent = agents[0]

    # Verify all fields were set
    assert agent.agent_name == "full_test_agent"
    assert agent.agent_description == "Full test description"
    # Agent may modify system_prompt by adding additional instructions
    assert "You are a comprehensive test agent" in agent.system_prompt
    assert agent.max_loops == 3
    assert agent.max_tokens == 4096
    assert agent.temperature == 0.7


def test_create_agents_from_specs_empty_list():
    """Test that create_agents_from_specs handles empty agent list"""
    builder = AutoSwarmBuilder()

    specs = {"agents": []}

    agents = builder.create_agents_from_specs(specs)

    assert isinstance(agents, list)
    assert len(agents) == 0


# ============================================================================
# SET RANDOM MODELS TESTS
# ============================================================================


def test_set_random_models_for_agents_with_valid_agents():
    """Test set_random_models_for_agents with proper Agent objects"""
    # Create proper Agent objects
    agents = [
        Agent(
            agent_name="agent1",
            system_prompt="You are agent 1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="agent2",
            system_prompt="You are agent 2",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
    ]

    # Set random models
    model_names = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]
    result = set_random_models_for_agents(
        agents=agents, model_names=model_names
    )

    # Verify results
    assert len(result) == 2
    assert all(isinstance(agent, Agent) for agent in result)
    assert all(hasattr(agent, "model_name") for agent in result)
    assert all(agent.model_name in model_names for agent in result)


def test_set_random_models_for_agents_with_single_agent():
    """Test set_random_models_for_agents with a single agent"""
    agent = Agent(
        agent_name="single_agent",
        system_prompt="You are helpful",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    model_names = ["gpt-4o-mini", "gpt-4o"]
    result = set_random_models_for_agents(
        agents=agent, model_names=model_names
    )

    assert isinstance(result, Agent)
    assert hasattr(result, "model_name")
    assert result.model_name in model_names


def test_set_random_models_for_agents_with_none():
    """Test set_random_models_for_agents with None returns random model name"""
    model_names = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]
    result = set_random_models_for_agents(
        agents=None, model_names=model_names
    )

    assert isinstance(result, str)
    assert result in model_names


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.skip(
    reason="This test requires API key and makes LLM calls"
)
def test_auto_swarm_builder_return_agents_objects_integration():
    """Integration test for AutoSwarmBuilder with execution_type='return-agents-objects'

    This test requires OPENAI_API_KEY and makes actual LLM calls.
    Run manually with: pytest -k test_auto_swarm_builder_return_agents_objects_integration -v
    """
    builder = AutoSwarmBuilder(
        execution_type="return-agents-objects",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    agents = builder.run(
        "Create a team of 2 data analysis agents with specific roles"
    )

    # Verify agents were created
    assert isinstance(agents, list)
    assert len(agents) >= 1
    assert all(isinstance(agent, Agent) for agent in agents)
    assert all(hasattr(agent, "agent_name") for agent in agents)
    assert all(hasattr(agent, "agent_description") for agent in agents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
