import pytest
from swarms import Agent, AgentRearrange


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_test_agent(name: str) -> Agent:
    """Create a real test agent"""
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Process tasks and respond concisely.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )


def create_test_agents():
    """Create multiple test agents"""
    return [
        create_test_agent("Agent1"),
        create_test_agent("Agent2"),
        create_test_agent("Agent3"),
    ]


# ============================================================================
# INITIALIZATION AND VALIDATION TESTS
# ============================================================================


def test_initialization_creates_agent_dict():
    """Test that agents list is converted to dictionary keyed by agent_name"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    assert isinstance(rearrange.agents, dict)
    assert "Agent1" in rearrange.agents
    assert "Agent2" in rearrange.agents
    assert "Agent3" in rearrange.agents
    assert len(rearrange.agents) == 3


def test_reliability_check_raises_on_empty_agents():
    """Test that ValueError is raised when agents list is empty"""
    with pytest.raises(ValueError, match="Agents list cannot be None or empty"):
        AgentRearrange(agents=[], flow="Agent1 -> Agent2")


def test_reliability_check_raises_on_zero_max_loops():
    """Test that ValueError is raised when max_loops is 0"""
    agents = create_test_agents()
    with pytest.raises(ValueError, match="max_loops cannot be 0"):
        AgentRearrange(
            agents=agents,
            flow="Agent1 -> Agent2",
            max_loops=0
        )


def test_reliability_check_raises_on_empty_flow():
    """Test that ValueError is raised when flow is empty"""
    agents = create_test_agents()
    with pytest.raises(ValueError, match="flow cannot be None or empty"):
        AgentRearrange(agents=agents, flow="")


def test_reliability_check_raises_on_none_flow():
    """Test that ValueError is raised when flow is None"""
    agents = create_test_agents()
    with pytest.raises(ValueError, match="flow cannot be None or empty"):
        AgentRearrange(agents=agents, flow=None)


def test_reliability_check_raises_on_empty_output_type():
    """Test that empty output_type raises ValueError"""
    agents = create_test_agents()
    with pytest.raises(ValueError, match="output_type cannot be None or empty"):
        AgentRearrange(
            agents=agents,
            flow="Agent1 -> Agent2",
            output_type=""
        )


def test_reliability_check_raises_on_none_output_type():
    """Test that None output_type raises ValueError"""
    agents = create_test_agents()
    with pytest.raises(ValueError, match="output_type cannot be None or empty"):
        AgentRearrange(
            agents=agents,
            flow="Agent1 -> Agent2",
            output_type=None
        )


# ============================================================================
# FLOW VALIDATION TESTS
# ============================================================================


def test_validate_flow_requires_arrow():
    """Test that flow must contain '->' separator"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 Agent2"
    )

    with pytest.raises(ValueError, match="Flow must include '->'"):
        rearrange.validate_flow()


def test_validate_flow_detects_unregistered_agent():
    """Test that flow validation catches agents not in agents dict"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent4"
    )

    with pytest.raises(ValueError, match="Agent 'Agent4' is not registered"):
        rearrange.validate_flow()


def test_validate_flow_allows_human_in_loop():
    """Test that 'H' is allowed in flow for human-in-the-loop"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> H -> Agent2"
    )

    assert rearrange.validate_flow() is True


def test_validate_flow_accepts_valid_sequential():
    """Test that valid sequential flow passes validation"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    assert rearrange.validate_flow() is True


def test_validate_flow_accepts_valid_concurrent():
    """Test that valid concurrent flow with commas passes validation"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2, Agent3"
    )

    assert rearrange.validate_flow() is True


def test_validate_flow_accepts_mixed_flow():
    """Test that mixed sequential and concurrent flow passes validation"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1, Agent2 -> Agent3"
    )

    assert rearrange.validate_flow() is True


def test_flow_with_whitespace():
    """Test that flow handles extra whitespace correctly"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="  Agent1   ->   Agent2  ,  Agent3  "
    )

    assert rearrange.validate_flow() is True


# ============================================================================
# AGENT MANAGEMENT TESTS
# ============================================================================


def test_add_agent_adds_to_dict():
    """Test that add_agent properly adds agent to agents dict"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    new_agent = create_test_agent("Agent4")
    rearrange.add_agent(new_agent)

    assert "Agent4" in rearrange.agents
    assert rearrange.agents["Agent4"] == new_agent
    assert len(rearrange.agents) == 4


def test_remove_agent_removes_from_dict():
    """Test that remove_agent properly removes agent from dict"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    rearrange.remove_agent("Agent2")

    assert "Agent2" not in rearrange.agents
    assert len(rearrange.agents) == 2
    assert "Agent1" in rearrange.agents
    assert "Agent3" in rearrange.agents


def test_add_agents_adds_multiple():
    """Test that add_agents adds multiple agents to dict"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    new_agents = [create_test_agent("Agent4"), create_test_agent("Agent5")]
    rearrange.add_agents(new_agents)

    assert "Agent4" in rearrange.agents
    assert "Agent5" in rearrange.agents
    assert len(rearrange.agents) == 5


# ============================================================================
# SEQUENTIAL AWARENESS TESTS
# ============================================================================


def test_get_agent_sequential_awareness_middle_agent():
    """Test that middle agent has awareness of both previous and next agents"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    awareness = rearrange.get_agent_sequential_awareness("Agent2")

    assert "Agent ahead: Agent1" in awareness
    assert "Agent behind: Agent3" in awareness


def test_get_agent_sequential_awareness_first_agent():
    """Test that first agent only has awareness of next agent"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    awareness = rearrange.get_agent_sequential_awareness("Agent1")

    assert "Agent behind: Agent2" in awareness
    assert "Agent ahead" not in awareness


def test_get_agent_sequential_awareness_last_agent():
    """Test that last agent only has awareness of previous agent"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    awareness = rearrange.get_agent_sequential_awareness("Agent3")

    assert "Agent ahead: Agent2" in awareness
    assert "Agent behind" not in awareness


def test_get_sequential_flow_structure():
    """Test that flow structure describes all steps with their relationships"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    structure = rearrange.get_sequential_flow_structure()

    assert "Sequential Flow Structure:" in structure
    assert "Step 1: Agent1" in structure
    assert "Step 2: Agent2" in structure
    assert "Step 3: Agent3" in structure
    assert "leads to: Agent2" in structure
    assert "follows: Agent1" in structure or "leads to: Agent3" in structure


def test_sequential_awareness_with_concurrent_agents():
    """Test awareness when multiple agents are in same step (concurrent)"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2, Agent3"
    )

    awareness2 = rearrange.get_agent_sequential_awareness("Agent2")
    awareness3 = rearrange.get_agent_sequential_awareness("Agent3")

    assert "Agent ahead: Agent1" in awareness2
    assert "Agent ahead: Agent1" in awareness3


def test_sequential_awareness_returns_empty_for_no_arrow():
    """Test that awareness returns empty string when no arrow in flow"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    rearrange.flow = "Agent1 Agent2"
    awareness = rearrange.get_agent_sequential_awareness("Agent1")

    assert awareness == ""


# ============================================================================
# FLOW MODIFICATION TESTS
# ============================================================================


def test_set_custom_flow_changes_flow():
    """Test that set_custom_flow properly updates the flow pattern"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2 -> Agent3"
    )

    rearrange.set_custom_flow("Agent3 -> Agent1")

    assert rearrange.flow == "Agent3 -> Agent1"
    assert rearrange.validate_flow() is True


# ============================================================================
# CONVERSATION TRACKING TESTS
# ============================================================================


def test_conversation_tracks_user_task():
    """Test that initial user task is added to conversation"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    rearrange.run("user task here")

    conversation_str = rearrange.conversation.return_history_as_string()
    assert "user task here" in conversation_str


def test_conversation_initialized_with_name():
    """Test that conversation is created with proper name"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        name="TestSwarm",
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    assert rearrange.conversation.name == "TestSwarm-Conversation"


def test_conversation_respects_time_enabled():
    """Test that conversation uses time_enabled parameter"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2",
        time_enabled=True
    )

    assert rearrange.conversation.time_enabled is True


def test_conversation_respects_message_id():
    """Test that conversation uses message_id_on parameter"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2",
        message_id_on=True
    )

    assert rearrange.conversation.message_id_on is True


def test_rules_added_to_conversation():
    """Test that rules parameter is added to conversation"""
    agents = create_test_agents()
    rules = "Always be concise and professional"
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2",
        rules=rules
    )

    conversation_str = rearrange.conversation.return_history_as_string()
    assert rules in conversation_str


# ============================================================================
# CALLABLE INTERFACE TESTS
# ============================================================================


def test_callable_interface_executes():
    """Test that AgentRearrange instance can be called directly"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    result = rearrange("test task")
    assert result is not None
    assert isinstance(result, str)


# ============================================================================
# EXECUTION TESTS
# ============================================================================


def test_run_returns_string():
    """Test that run returns a string result"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    result = rearrange.run("test task")
    assert isinstance(result, str)
    assert len(result) > 0


def test_batch_run_processes_all_tasks():
    """Test that batch_run processes all tasks"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    tasks = ["task1", "task2", "task3"]
    results = rearrange.batch_run(tasks, batch_size=2)

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_concurrent_run_processes_all_tasks():
    """Test that concurrent_run processes all tasks"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    tasks = ["task1", "task2"]
    results = rearrange.concurrent_run(tasks, max_workers=2)

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


def test_to_dict_returns_dict():
    """Test that to_dict returns a dictionary"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    result = rearrange.to_dict()
    assert isinstance(result, dict)


def test_to_dict_contains_key_attributes():
    """Test that to_dict contains essential attributes"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        name="TestSwarm",
        description="Test Description",
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    result = rearrange.to_dict()
    assert "name" in result
    assert "description" in result
    assert "flow" in result
    assert result["name"] == "TestSwarm"
    assert result["description"] == "Test Description"
    assert result["flow"] == "Agent1 -> Agent2"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_initialization_with_custom_name():
    """Test initialization with custom name"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        name="CustomSwarm",
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    assert rearrange.name == "CustomSwarm"


def test_initialization_with_custom_description():
    """Test initialization with custom description"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        description="Custom Description",
        agents=agents,
        flow="Agent1 -> Agent2"
    )

    assert rearrange.description == "Custom Description"


def test_initialization_with_max_loops():
    """Test initialization with custom max_loops"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2",
        max_loops=5
    )

    assert rearrange.max_loops == 5


def test_max_loops_default_to_one_if_zero():
    """Test that max_loops defaults to 1 if set to 0 in constructor"""
    agents = create_test_agents()
    rearrange = AgentRearrange(
        agents=agents,
        flow="Agent1 -> Agent2",
        max_loops=-1
    )

    assert rearrange.max_loops == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
