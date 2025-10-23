import pytest

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


def create_test_agent(name: str, description: str = None) -> Agent:
    """Helper function to create a test agent"""
    return Agent(
        agent_name=name,
        description=description or f"Agent specialized in {name} tasks",
        system_prompt=f"You are {name}, a helpful assistant.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_multi_agent_router_initialization():
    """Test MultiAgentRouter basic initialization"""
    agents = [
        create_test_agent("TestAgent1"),
        create_test_agent("TestAgent2"),
    ]
    router = MultiAgentRouter(agents=agents)

    assert router.name == "swarm-router"
    assert len(router.agents) == 2
    assert "TestAgent1" in router.agents
    assert "TestAgent2" in router.agents


def test_multi_agent_router_with_custom_name():
    """Test MultiAgentRouter with custom name"""
    agents = [create_test_agent("TestAgent1")]
    router = MultiAgentRouter(agents=agents, name="CustomRouter")

    assert router.name == "CustomRouter"


def test_multi_agent_router_with_custom_description():
    """Test MultiAgentRouter with custom description"""
    agents = [create_test_agent("TestAgent1")]
    custom_desc = "Custom router description"
    router = MultiAgentRouter(agents=agents, description=custom_desc)

    assert router.description == custom_desc


def test_multi_agent_router_agents_dict():
    """Test that agents are stored as dictionary with agent_name as key"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")
    router = MultiAgentRouter(agents=[agent1, agent2])

    assert isinstance(router.agents, dict)
    assert router.agents["Agent1"] == agent1
    assert router.agents["Agent2"] == agent2


def test_multi_agent_router_conversation_initialized():
    """Test that conversation object is initialized"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    assert hasattr(router, 'conversation')
    assert router.conversation is not None


def test_multi_agent_router_model_configuration():
    """Test MultiAgentRouter with custom model configuration"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(
        agents=agents,
        model="gpt-4o",
        temperature=0.5
    )

    assert router.model == "gpt-4o"
    assert router.temperature == 0.5


def test_multi_agent_router_output_type():
    """Test MultiAgentRouter with different output types"""
    agents = [create_test_agent("TestAgent")]

    for output_type in ["dict", "string", "list"]:
        router = MultiAgentRouter(agents=agents, output_type=output_type)
        assert router.output_type == output_type


def test_multi_agent_router_skip_null_tasks():
    """Test skip_null_tasks configuration"""
    agents = [create_test_agent("TestAgent")]

    router1 = MultiAgentRouter(agents=agents, skip_null_tasks=True)
    assert router1.skip_null_tasks is True

    router2 = MultiAgentRouter(agents=agents, skip_null_tasks=False)
    assert router2.skip_null_tasks is False


# ============================================================================
# SYSTEM PROMPT TESTS
# ============================================================================


def test_boss_system_prompt_creation():
    """Test boss system prompt generation"""
    agents = [
        create_test_agent("Agent1", "Specialist in coding"),
        create_test_agent("Agent2", "Specialist in writing"),
    ]
    router = MultiAgentRouter(agents=agents)
    prompt = router._create_boss_system_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Agent1" in prompt
    assert "Agent2" in prompt


def test_boss_system_prompt_contains_routing_instructions():
    """Test that boss system prompt contains routing logic"""
    agents = [create_test_agent("Agent1")]
    router = MultiAgentRouter(agents=agents)
    prompt = router._create_boss_system_prompt()

    assert "intelligent boss agent" in prompt
    assert "routing" in prompt.lower() or "route" in prompt.lower()


def test_boss_system_prompt_includes_agent_descriptions():
    """Test that system prompt includes agent descriptions"""
    agents = [
        create_test_agent("CodeAgent", "Expert in writing code"),
        create_test_agent("WritingAgent", "Expert in creative writing"),
    ]
    router = MultiAgentRouter(agents=agents)
    prompt = router._create_boss_system_prompt()

    assert "CodeAgent" in prompt
    assert "WritingAgent" in prompt
    assert "Expert in writing code" in prompt
    assert "Expert in creative writing" in prompt


def test_boss_system_prompt_with_custom_system_prompt():
    """Test router with custom system prompt prepended"""
    agents = [create_test_agent("Agent1")]
    custom_prompt = "Custom routing instructions: Be very careful."
    router = MultiAgentRouter(agents=agents, system_prompt=custom_prompt)

    # The function_caller should have the combined prompt
    assert router.function_caller is not None


# ============================================================================
# AGENT FINDING TESTS
# ============================================================================


def test_agents_stored_by_name():
    """Test that agents can be accessed by name"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")
    router = MultiAgentRouter(agents=[agent1, agent2])

    assert "Agent1" in router.agents
    assert "Agent2" in router.agents
    assert router.agents["Agent1"].agent_name == "Agent1"
    assert router.agents["Agent2"].agent_name == "Agent2"


def test_find_nonexistent_agent():
    """Test checking for nonexistent agent"""
    agent1 = create_test_agent("Agent1")
    router = MultiAgentRouter(agents=[agent1])

    assert "NonexistentAgent" not in router.agents


def test_multiple_agents_accessible():
    """Test that all agents are accessible in agents dict"""
    agents = [create_test_agent(f"Agent{i}") for i in range(5)]
    router = MultiAgentRouter(agents=agents)

    assert len(router.agents) == 5
    for i in range(5):
        assert f"Agent{i}" in router.agents


# ============================================================================
# CALLABLE INTERFACE TESTS
# ============================================================================


def test_router_repr():
    """Test __repr__ method"""
    agents = [create_test_agent("Agent1"), create_test_agent("Agent2")]
    router = MultiAgentRouter(agents=agents)

    repr_str = repr(router)
    assert "MultiAgentRouter" in repr_str
    assert "swarm-router" in repr_str


def test_router_callable_interface():
    """Test that router can be called directly"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents, print_on=False)

    # Router should be callable
    assert callable(router)


def test_run_method_exists():
    """Test that run method exists and is callable"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    assert hasattr(router, 'run')
    assert callable(router.run)


# ============================================================================
# BATCH ROUTING TESTS
# ============================================================================


def test_batch_run_method_exists():
    """Test that batch_run method exists"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    assert hasattr(router, 'batch_run')
    assert callable(router.batch_run)


def test_batch_run_returns_list():
    """Test that batch_run returns a list"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents, print_on=False)

    tasks = ["Task 1", "Task 2"]
    results = router.batch_run(tasks)

    assert isinstance(results, list)
    assert len(results) == 2


def test_concurrent_batch_run_method_exists():
    """Test that concurrent_batch_run method exists"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    assert hasattr(router, 'concurrent_batch_run')
    assert callable(router.concurrent_batch_run)


def test_concurrent_batch_run_returns_list():
    """Test that concurrent_batch_run returns a list"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents, print_on=False)

    tasks = ["Task 1", "Task 2"]
    results = router.concurrent_batch_run(tasks)

    assert isinstance(results, list)
    assert len(results) == 2


def test_batch_run_with_empty_list():
    """Test batch_run with empty task list"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    results = router.batch_run([])
    assert isinstance(results, list)
    assert len(results) == 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_route_task_with_no_agents_raises_error():
    """Test that routing with no agents raises an error"""
    router = MultiAgentRouter(agents=[])

    with pytest.raises(Exception):
        router.route_task("Test task")


def test_handle_single_handoff_validates_agent_name():
    """Test that handle_single_handoff validates agent exists"""
    agents = [create_test_agent("Agent1")]
    router = MultiAgentRouter(agents=agents)

    invalid_handoff = {
        "handoffs": [{
            "agent_name": "NonexistentAgent",
            "reasoning": "Test",
            "task": "Test task"
        }]
    }

    with pytest.raises(ValueError, match="unknown agent"):
        router.handle_single_handoff(invalid_handoff, "Test task")


def test_handle_multiple_handoffs_validates_agent_names():
    """Test that handle_multiple_handoffs validates all agent names"""
    agents = [create_test_agent("Agent1")]
    router = MultiAgentRouter(agents=agents)

    invalid_handoff = {
        "handoffs": [
            {
                "agent_name": "Agent1",
                "reasoning": "Test",
                "task": "Task 1"
            },
            {
                "agent_name": "NonexistentAgent",
                "reasoning": "Test",
                "task": "Task 2"
            }
        ]
    }

    with pytest.raises(ValueError, match="unknown agent"):
        router.handle_multiple_handoffs(invalid_handoff, "Test task")


# ============================================================================
# CONFIGURATION VALIDATION TESTS
# ============================================================================


def test_router_with_single_agent():
    """Test router with only one agent"""
    agent = create_test_agent("OnlyAgent")
    router = MultiAgentRouter(agents=[agent])

    assert len(router.agents) == 1
    assert "OnlyAgent" in router.agents


def test_router_with_many_agents():
    """Test router with many agents"""
    agents = [create_test_agent(f"Agent{i}") for i in range(10)]
    router = MultiAgentRouter(agents=agents)

    assert len(router.agents) == 10


def test_router_print_on_configuration():
    """Test print_on configuration"""
    agents = [create_test_agent("TestAgent")]

    router1 = MultiAgentRouter(agents=agents, print_on=True)
    assert router1.print_on is True

    router2 = MultiAgentRouter(agents=agents, print_on=False)
    assert router2.print_on is False


def test_function_caller_initialized():
    """Test that function_caller is initialized with LiteLLM"""
    agents = [create_test_agent("TestAgent")]
    router = MultiAgentRouter(agents=agents)

    assert hasattr(router, 'function_caller')
    assert router.function_caller is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_router_initialization_with_all_parameters():
    """Test complete router initialization with all parameters"""
    agents = [
        create_test_agent("Agent1", "First agent"),
        create_test_agent("Agent2", "Second agent"),
    ]

    router = MultiAgentRouter(
        name="CompleteRouter",
        description="A fully configured router",
        agents=agents,
        model="gpt-4o-mini",
        temperature=0.2,
        output_type="string",
        print_on=False,
        skip_null_tasks=True
    )

    assert router.name == "CompleteRouter"
    assert router.description == "A fully configured router"
    assert len(router.agents) == 2
    assert router.model == "gpt-4o-mini"
    assert router.temperature == 0.2
    assert router.output_type == "string"
    assert router.print_on is False
    assert router.skip_null_tasks is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
