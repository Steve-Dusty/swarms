import pytest
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
)
from swarms.structs.agent import Agent


def create_sample_agents():
    """Create sample real agents for testing."""
    agents = []
    for i in range(5):
        agent = Agent(
            agent_name=f"Board-Member-{i+1}",
            agent_description=f"Board member {i+1} with expertise in strategic decision making",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        )
        agents.append(agent)
    return agents


def create_basic_board_swarm():
    """Create a basic Board of Directors swarm for testing."""
    sample_agents = create_sample_agents()
    return BoardOfDirectorsSwarm(
        name="Test-Board-Swarm",
        description="Test board of directors swarm for comprehensive testing",
        agents=sample_agents,
        max_loops=1,
        verbose=False,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_board_of_directors_swarm_basic_initialization():
    """Test basic BoardOfDirectorsSwarm initialization with multiple agents"""
    basic_board_swarm = create_basic_board_swarm()

    assert basic_board_swarm.name == "Test-Board-Swarm"
    assert (
        basic_board_swarm.description
        == "Test board of directors swarm for comprehensive testing"
    )
    assert len(basic_board_swarm.agents) == 5
    assert basic_board_swarm.max_loops == 1
    assert basic_board_swarm.verbose is False
    assert basic_board_swarm.board_model_name == "gpt-4o-mini"
    assert basic_board_swarm.decision_threshold == 0.6
    assert basic_board_swarm.enable_voting is True
    assert basic_board_swarm.enable_consensus is True


def test_board_initialization_creates_default_board_members():
    """Test that board initialization creates default board members"""
    agents = create_sample_agents()
    swarm = BoardOfDirectorsSwarm(
        agents=agents,
        max_loops=1,
        verbose=False
    )

    assert len(swarm.board_members) == 3
    assert any(member.agent.agent_name == "Chairman" for member in swarm.board_members)
    assert any(member.agent.agent_name == "Vice-Chairman" for member in swarm.board_members)
    assert any(member.agent.agent_name == "Secretary" for member in swarm.board_members)


def test_board_initialization_with_custom_config():
    """Test BoardOfDirectorsSwarm with custom configuration"""
    agents = create_sample_agents()

    swarm = BoardOfDirectorsSwarm(
        name="CustomBoard",
        description="Custom board description",
        agents=agents,
        max_loops=3,
        decision_threshold=0.75,
        enable_voting=False,
        enable_consensus=True,
        verbose=True
    )

    assert swarm.name == "CustomBoard"
    assert swarm.description == "Custom board description"
    assert swarm.max_loops == 3
    assert swarm.decision_threshold == 0.75
    assert swarm.enable_voting is False
    assert swarm.enable_consensus is True
    assert swarm.verbose is True


# ============================================================================
# VALIDATION TESTS
# ============================================================================


def test_board_raises_on_empty_agents_list():
    """Test that empty agents list raises ValueError"""
    with pytest.raises(ValueError, match="No agents found|agents"):
        BoardOfDirectorsSwarm(agents=[])


def test_board_raises_on_zero_max_loops():
    """Test that max_loops=0 raises ValueError"""
    agents = create_sample_agents()

    with pytest.raises(ValueError, match="Max loops|greater than 0"):
        BoardOfDirectorsSwarm(agents=agents, max_loops=0)


def test_board_raises_on_negative_max_loops():
    """Test that negative max_loops raises ValueError"""
    agents = create_sample_agents()

    with pytest.raises(ValueError, match="Max loops|greater than 0"):
        BoardOfDirectorsSwarm(agents=agents, max_loops=-1)


def test_board_raises_on_invalid_decision_threshold():
    """Test that decision threshold must be between 0.0 and 1.0"""
    agents = create_sample_agents()

    with pytest.raises(ValueError, match="Decision threshold|between"):
        BoardOfDirectorsSwarm(
            agents=agents,
            decision_threshold=1.5
        )


def test_board_raises_on_negative_decision_threshold():
    """Test that negative decision threshold raises ValueError"""
    agents = create_sample_agents()

    with pytest.raises(ValueError, match="Decision threshold|between"):
        BoardOfDirectorsSwarm(
            agents=agents,
            decision_threshold=-0.1
        )


# ============================================================================
# CONVERSATION MANAGEMENT TESTS
# ============================================================================


def test_board_initializes_conversation():
    """Test that board creates conversation object"""
    swarm = create_basic_board_swarm()

    assert hasattr(swarm, "conversation")
    assert swarm.conversation is not None


def test_conversation_time_enabled_setting():
    """Test that conversation has time_enabled=False"""
    swarm = create_basic_board_swarm()

    assert swarm.conversation.time_enabled is False


# ============================================================================
# EXECUTION TESTS
# ============================================================================


def test_board_run_returns_string():
    """Test that board run returns string output"""
    swarm = create_basic_board_swarm()

    result = swarm.run("Create a strategy for market expansion")

    assert isinstance(result, str)
    assert len(result) > 0


def test_board_run_with_simple_task():
    """Test board execution with simple task"""
    swarm = create_basic_board_swarm()

    result = swarm.run("Summarize the key points of this proposal")

    assert result is not None
    assert isinstance(result, (str, dict, list))


# ============================================================================
# BOARD MEMBER MANAGEMENT TESTS
# ============================================================================


def test_add_board_member():
    """Test adding a new board member"""
    from swarms.structs.board_of_directors_swarm import BoardMember, BoardMemberRole

    swarm = create_basic_board_swarm()
    initial_count = len(swarm.board_members)

    new_agent = Agent(
        agent_name="New-Treasurer",
        agent_description="Treasurer with financial expertise",
        model_name="gpt-4o-mini",
        max_loops=1
    )

    new_member = BoardMember(
        agent=new_agent,
        role=BoardMemberRole.TREASURER,
        voting_weight=1.0,
        expertise_areas=["finance", "accounting"]
    )

    swarm.add_board_member(new_member)

    assert len(swarm.board_members) == initial_count + 1
    assert swarm.get_board_member("New-Treasurer") is not None


def test_remove_board_member():
    """Test removing a board member"""
    swarm = create_basic_board_swarm()
    initial_count = len(swarm.board_members)

    swarm.remove_board_member("Secretary")

    assert len(swarm.board_members) == initial_count - 1
    assert swarm.get_board_member("Secretary") is None


def test_get_board_member_existing():
    """Test getting an existing board member"""
    swarm = create_basic_board_swarm()

    chairman = swarm.get_board_member("Chairman")

    assert chairman is not None
    assert chairman.agent.agent_name == "Chairman"


def test_get_board_member_nonexistent():
    """Test getting a nonexistent board member returns None"""
    swarm = create_basic_board_swarm()

    result = swarm.get_board_member("NonExistent")

    assert result is None


def test_get_board_summary():
    """Test getting board summary"""
    swarm = create_basic_board_swarm()

    summary = swarm.get_board_summary()

    assert isinstance(summary, dict)
    assert "board_name" in summary
    assert "total_members" in summary
    assert "members" in summary
    assert "total_agents" in summary
    assert summary["board_name"] == "Test-Board-Swarm"
    assert summary["total_members"] == 3
    assert summary["total_agents"] == 5


def test_board_summary_contains_member_details():
    """Test that board summary includes member details"""
    swarm = create_basic_board_swarm()

    summary = swarm.get_board_summary()

    assert len(summary["members"]) == 3
    for member in summary["members"]:
        assert "name" in member
        assert "role" in member
        assert "voting_weight" in member
        assert "expertise_areas" in member


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_board_with_specialized_roles():
    """Test board with specialized C-suite roles"""
    ceo = Agent(
        agent_name="CEO",
        agent_description="Chief Executive Officer",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False
    )

    cfo = Agent(
        agent_name="CFO",
        agent_description="Chief Financial Officer",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False
    )

    cto = Agent(
        agent_name="CTO",
        agent_description="Chief Technology Officer",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False
    )

    swarm = BoardOfDirectorsSwarm(
        name="Executive-Board",
        agents=[ceo, cfo, cto],
        max_loops=1,
        verbose=False
    )

    assert len(swarm.agents) == 3
    assert swarm.name == "Executive-Board"


def test_board_with_multiple_loops():
    """Test board execution with multiple loops"""
    swarm = create_basic_board_swarm()
    swarm.max_loops = 2

    result = swarm.run("Plan quarterly goals")

    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])