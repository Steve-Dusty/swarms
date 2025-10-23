import pytest

from swarms.structs.agent import Agent
from swarms.structs.groupchat import (
    GroupChat,
    round_robin,
    expertise_based,
    random_selection,
    sentiment_based,
    length_based,
    question_based,
    topic_based,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_test_agents(num_agents):
    """Create test agents with diverse specializations"""
    specialties = [
        (
            "Finance",
            "You are a financial expert focusing on investment strategies and market analysis.",
        ),
        (
            "Tech",
            "You are a technology expert specializing in AI and cybersecurity.",
        ),
        (
            "Healthcare",
            "You are a healthcare professional with expertise in public health.",
        ),
        (
            "Marketing",
            "You are a marketing strategist focusing on digital trends.",
        ),
        (
            "Legal",
            "You are a legal expert specializing in corporate law.",
        ),
    ]

    agents = []
    for i in range(num_agents):
        specialty, prompt = specialties[i % len(specialties)]
        agents.append(
            Agent(
                agent_name=f"{specialty}-Agent-{i+1}",
                system_prompt=prompt,
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            )
        )
    return agents


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_groupchat_initialization_basic():
    """Test basic GroupChat initialization"""
    agents = create_test_agents(2)
    chat = GroupChat(
        name="Test Chat",
        description="A test group chat",
        agents=agents,
        max_loops=2,
    )

    assert chat.name == "Test Chat"
    assert chat.description == "A test group chat"
    assert len(chat.agents) == 2
    assert chat.max_loops == 2


def test_groupchat_initialization_with_rules():
    """Test GroupChat initialization with conversation rules"""
    agents = create_test_agents(3)
    rules = "1. Be professional\n2. Stay on topic\n3. Be concise"

    chat = GroupChat(
        name="Rules Test",
        description="Chat with rules",
        agents=agents,
        rules=rules,
        max_loops=1,
    )

    assert chat.rules == rules
    assert len(chat.agents) == 3


# ============================================================================
# EXECUTION TESTS
# ============================================================================


def test_groupchat_basic_run():
    """Test basic GroupChat conversation execution"""
    agents = create_test_agents(2)
    chat = GroupChat(
        name="Basic Test",
        description="Basic conversation test",
        agents=agents,
        max_loops=1,
    )

    result = chat.run("Introduce yourself briefly")
    assert result is not None


# ============================================================================
# SPEAKER FUNCTION TESTS
# ============================================================================


def test_groupchat_round_robin_speaker():
    """Test GroupChat with round_robin speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Round Robin Test",
        agents=agents,
        speaker_fn=round_robin,
        max_loops=1,
    )

    result = chat.run("Share your expertise")
    assert result is not None


def test_groupchat_expertise_based_speaker():
    """Test GroupChat with expertise_based speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Expertise Based Test",
        agents=agents,
        speaker_fn=expertise_based,
        max_loops=1,
    )

    result = chat.run("Discuss AI impact on your field")
    assert result is not None


def test_groupchat_random_selection_speaker():
    """Test GroupChat with random_selection speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Random Selection Test",
        agents=agents,
        speaker_fn=random_selection,
        max_loops=1,
    )

    result = chat.run("What are your thoughts?")
    assert result is not None


def test_groupchat_sentiment_based_speaker():
    """Test GroupChat with sentiment_based speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Sentiment Based Test",
        agents=agents,
        speaker_fn=sentiment_based,
        max_loops=1,
    )

    result = chat.run("Share positive insights")
    assert result is not None


def test_groupchat_length_based_speaker():
    """Test GroupChat with length_based speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Length Based Test",
        agents=agents,
        speaker_fn=length_based,
        max_loops=1,
    )

    result = chat.run("Provide detailed analysis")
    assert result is not None


def test_groupchat_question_based_speaker():
    """Test GroupChat with question_based speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Question Based Test",
        agents=agents,
        speaker_fn=question_based,
        max_loops=1,
    )

    result = chat.run("What challenges do you see?")
    assert result is not None


def test_groupchat_topic_based_speaker():
    """Test GroupChat with topic_based speaker function"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Topic Based Test",
        agents=agents,
        speaker_fn=topic_based,
        max_loops=1,
    )

    result = chat.run("Discuss digital transformation")
    assert result is not None


# ============================================================================
# AGENT COUNT VARIATION TESTS
# ============================================================================


def test_groupchat_minimum_agents_requirement():
    """Test GroupChat requires at least 2 agents"""
    agents = create_test_agents(1)

    # Should raise ValueError with less than 2 agents
    with pytest.raises(ValueError, match="At least two agents are required"):
        GroupChat(
            name="Single Agent Test",
            agents=agents,
            max_loops=1,
        )


def test_groupchat_multiple_agents():
    """Test GroupChat with multiple agents"""
    agents = create_test_agents(5)
    chat = GroupChat(
        name="Multiple Agents Test",
        agents=agents,
        max_loops=1,
    )

    result = chat.run("Share your perspectives")
    assert result is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_groupchat_empty_agents_list():
    """Test GroupChat with empty agents list raises error"""
    with pytest.raises(ValueError):
        GroupChat(agents=[])


def test_groupchat_invalid_max_loops():
    """Test GroupChat with invalid max_loops raises error"""
    agents = create_test_agents(1)
    with pytest.raises(ValueError):
        GroupChat(agents=agents, max_loops=0)


def test_groupchat_empty_task():
    """Test GroupChat with empty task"""
    agents = create_test_agents(2)  # GroupChat requires at least 2 agents
    chat = GroupChat(agents=agents)

    # Empty task should either raise error or handle gracefully
    try:
        result = chat.run("")
        # If it doesn't raise, just verify it returns something
        assert result is not None or result == ""
    except ValueError:
        # This is also acceptable
        pass


def test_groupchat_none_task():
    """Test GroupChat with None task"""
    agents = create_test_agents(2)  # GroupChat requires at least 2 agents
    chat = GroupChat(agents=agents)

    # None task should raise TypeError
    with pytest.raises((ValueError, TypeError)):
        chat.run(None)


# ============================================================================
# CONCURRENT EXECUTION TESTS
# ============================================================================


def test_groupchat_concurrent_run():
    """Test concurrent execution of multiple tasks"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Concurrent Test",
        agents=agents,
        max_loops=1,
    )

    tasks = [
        "Task 1: Introduce yourself",
        "Task 2: What's your specialty?",
        "Task 3: How can you help?",
    ]

    results = chat.concurrent_run(tasks)
    assert results is not None
    assert len(results) == len(tasks)


# ============================================================================
# CONVERSATION WITH RULES TESTS
# ============================================================================


def test_groupchat_with_conversation_rules():
    """Test GroupChat with specific conversation rules"""
    agents = create_test_agents(3)
    rules = """
    1. Keep responses under 50 words
    2. Be professional
    3. Stay on topic
    4. Provide unique perspectives
    """

    chat = GroupChat(
        name="Rules Test",
        description="Testing with specific rules",
        agents=agents,
        max_loops=1,
        rules=rules,
    )

    result = chat.run("How can we ensure ethical AI development?")
    assert result is not None


# ============================================================================
# LOOP VARIATION TESTS
# ============================================================================


def test_groupchat_multiple_loops():
    """Test GroupChat with multiple conversation loops"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Multiple Loops Test",
        agents=agents,
        max_loops=3,
    )

    result = chat.run("Discuss future trends in your field")
    assert result is not None


def test_groupchat_configuration_options():
    """Test GroupChat with various configuration options"""
    agents = create_test_agents(3)
    chat = GroupChat(
        name="Config Test",
        description="Testing configuration",
        agents=agents,
        max_loops=2,
        speaker_fn=round_robin,
        rules="Be concise and professional",
    )

    assert chat.name == "Config Test"
    assert chat.description == "Testing configuration"
    assert chat.max_loops == 2
    assert chat.speaker_fn == round_robin
    assert chat.rules == "Be concise and professional"
