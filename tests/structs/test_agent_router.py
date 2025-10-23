import pytest
from swarms.structs.agent_router import AgentRouter
from swarms.structs.agent import Agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_test_agent(name: str, description: str, system_prompt: str) -> Agent:
    """Create a real test agent"""
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_initialization_default_parameters():
    """Test AgentRouter initialization with defaults"""
    router = AgentRouter()

    assert router.embedding_model == "text-embedding-ada-002"
    assert router.n_agents == 1
    assert router.api_key is None
    assert router.api_base is None
    assert router.agents == []
    assert router.agent_embeddings == []
    assert router.agent_metadata == []


def test_initialization_custom_embedding_model():
    """Test initialization with custom embedding model"""
    router = AgentRouter(embedding_model="text-embedding-3-large")

    assert router.embedding_model == "text-embedding-3-large"


def test_initialization_custom_n_agents():
    """Test initialization with custom n_agents"""
    router = AgentRouter(n_agents=5)

    assert router.n_agents == 5


def test_initialization_with_api_credentials():
    """Test initialization with API key and base"""
    router = AgentRouter(
        api_key="test_key",
        api_base="https://custom.api.base"
    )

    assert router.api_key == "test_key"
    assert router.api_base == "https://custom.api.base"


def test_initialization_with_agents():
    """Test that agents provided during initialization are added"""
    agent1 = create_test_agent("Agent1", "First agent", "Prompt 1")
    agent2 = create_test_agent("Agent2", "Second agent", "Prompt 2")

    router = AgentRouter(agents=[agent1, agent2])

    assert len(router.agents) == 2
    assert len(router.agent_embeddings) == 2
    assert len(router.agent_metadata) == 2


# ============================================================================
# COSINE SIMILARITY TESTS
# ============================================================================


def test_cosine_similarity_identical_vectors():
    """Test that identical vectors have similarity of 1.0"""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)

    assert result == 1.0


def test_cosine_similarity_orthogonal_vectors():
    """Test that perpendicular vectors have similarity of 0.0"""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)

    assert result == 0.0


def test_cosine_similarity_opposite_vectors():
    """Test that opposite vectors have similarity of -1.0"""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)

    assert result == -1.0


def test_cosine_similarity_45_degree_vectors():
    """Test cosine similarity at 45 degrees"""
    router = AgentRouter()
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 1.0]

    result = router._cosine_similarity(vec1, vec2)

    assert abs(result - 0.7071067811865475) < 1e-10


def test_cosine_similarity_different_magnitudes():
    """Test that similarity is independent of vector magnitude"""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [5.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)

    assert result == 1.0


def test_cosine_similarity_zero_vector():
    """Test that zero vector returns 0.0 similarity"""
    router = AgentRouter()
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [0.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)

    assert result == 0.0


def test_cosine_similarity_different_lengths_raises():
    """Test that vectors of different lengths raise ValueError"""
    router = AgentRouter()
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="Vectors must have the same length"):
        router._cosine_similarity(vec1, vec2)


# ============================================================================
# AGENT MANAGEMENT TESTS
# ============================================================================


def test_add_agent_adds_to_lists():
    """Test that add_agent adds agent to all three lists"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "A test agent", "Test prompt")

    router.add_agent(agent)

    assert len(router.agents) == 1
    assert len(router.agent_embeddings) == 1
    assert len(router.agent_metadata) == 1


def test_add_agent_stores_correct_agent():
    """Test that added agent is stored correctly"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "A test agent", "Test prompt")

    router.add_agent(agent)

    assert router.agents[0] == agent


def test_add_agent_metadata_structure():
    """Test that agent metadata has correct structure"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "A test agent", "Test prompt")

    router.add_agent(agent)

    metadata = router.agent_metadata[0]
    assert "name" in metadata
    assert "text" in metadata
    assert metadata["name"] == "TestAgent"
    assert "TestAgent" in metadata["text"]
    assert "A test agent" in metadata["text"]
    assert "Test prompt" in metadata["text"]


def test_add_agent_embedding_is_list():
    """Test that generated embedding is a list of floats"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "A test agent", "Test prompt")

    router.add_agent(agent)

    embedding = router.agent_embeddings[0]
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


def test_add_agents_multiple():
    """Test adding multiple agents"""
    router = AgentRouter()
    agent1 = create_test_agent("Agent1", "First agent", "Prompt 1")
    agent2 = create_test_agent("Agent2", "Second agent", "Prompt 2")
    agent3 = create_test_agent("Agent3", "Third agent", "Prompt 3")

    router.add_agents([agent1, agent2, agent3])

    assert len(router.agents) == 3
    assert len(router.agent_embeddings) == 3
    assert len(router.agent_metadata) == 3


def test_add_agents_maintains_order():
    """Test that agents are stored in the order they are added"""
    router = AgentRouter()
    agent1 = create_test_agent("Agent1", "First", "P1")
    agent2 = create_test_agent("Agent2", "Second", "P2")

    router.add_agents([agent1, agent2])

    assert router.agents[0].agent_name == "Agent1"
    assert router.agents[1].agent_name == "Agent2"
    assert router.agent_metadata[0]["name"] == "Agent1"
    assert router.agent_metadata[1]["name"] == "Agent2"


# ============================================================================
# FIND BEST AGENT TESTS
# ============================================================================


def test_find_best_agent_no_agents_returns_none():
    """Test that finding best agent with no agents returns None"""
    router = AgentRouter()

    result = router.find_best_agent("test task")

    assert result is None


def test_find_best_agent_single_agent():
    """Test that single agent is returned as best"""
    router = AgentRouter()
    agent = create_test_agent("OnlyAgent", "The only agent", "Do tasks")

    router.add_agent(agent)
    result = router.find_best_agent("perform a task")

    assert result == agent


def test_find_best_agent_returns_agent_type():
    """Test that find_best_agent returns an Agent instance"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "Test agent", "Test prompt")

    router.add_agent(agent)
    result = router.find_best_agent("test task")

    assert isinstance(result, Agent)


def test_find_best_agent_semantic_matching():
    """Test that semantically similar agent is selected"""
    router = AgentRouter()

    financial_agent = create_test_agent(
        "FinancialAnalyst",
        "Expert in financial analysis and accounting",
        "Analyze financial statements and provide insights"
    )
    marketing_agent = create_test_agent(
        "MarketingExpert",
        "Expert in marketing and brand strategy",
        "Create marketing campaigns and analyze brand positioning"
    )

    router.add_agents([financial_agent, marketing_agent])

    result = router.find_best_agent("Analyze the company's balance sheet")

    assert result == financial_agent


def test_find_best_agent_with_multiple_agents():
    """Test finding best agent among multiple options"""
    router = AgentRouter()

    agent1 = create_test_agent("Agent1", "General assistant", "Help with tasks")
    agent2 = create_test_agent("Agent2", "Code specialist", "Write and debug code")
    agent3 = create_test_agent("Agent3", "Data analyst", "Analyze data and create reports")

    router.add_agents([agent1, agent2, agent3])

    result = router.find_best_agent("Write a Python function")

    assert result is not None
    assert result in [agent1, agent2, agent3]


# ============================================================================
# UPDATE AGENT HISTORY TESTS
# ============================================================================


def test_update_agent_history_updates_embedding():
    """Test that updating history regenerates agent embedding"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "Test agent", "Test prompt")

    router.add_agent(agent)
    original_embedding = router.agent_embeddings[0].copy()

    agent.short_memory.add("user", "test message")
    router.update_agent_history("TestAgent")

    updated_embedding = router.agent_embeddings[0]
    assert updated_embedding != original_embedding


def test_update_agent_history_updates_metadata():
    """Test that updating history updates agent metadata"""
    router = AgentRouter()
    agent = create_test_agent("TestAgent", "Test agent", "Test prompt")

    router.add_agent(agent)
    original_text = router.agent_metadata[0]["text"]

    agent.short_memory.add("user", "new conversation")
    router.update_agent_history("TestAgent")

    updated_text = router.agent_metadata[0]["text"]
    assert len(updated_text) > len(original_text)


def test_update_agent_history_nonexistent_agent():
    """Test that updating nonexistent agent doesn't raise exception"""
    router = AgentRouter()
    agent = create_test_agent("RealAgent", "Real agent", "Real prompt")

    router.add_agent(agent)
    router.update_agent_history("FakeAgent")

    assert len(router.agents) == 1


def test_update_agent_history_maintains_agent_count():
    """Test that update doesn't change number of agents"""
    router = AgentRouter()
    agent1 = create_test_agent("Agent1", "First", "P1")
    agent2 = create_test_agent("Agent2", "Second", "P2")

    router.add_agents([agent1, agent2])
    router.update_agent_history("Agent1")

    assert len(router.agents) == 2
    assert len(router.agent_embeddings) == 2
    assert len(router.agent_metadata) == 2


# ============================================================================
# EDGE CASES
# ============================================================================


def test_empty_task_string():
    """Test behavior with empty task string"""
    router = AgentRouter()
    agent = create_test_agent("Agent", "Description", "Prompt")

    router.add_agent(agent)
    result = router.find_best_agent("")

    assert result is not None


def test_very_long_task_description():
    """Test with very long task description"""
    router = AgentRouter()
    agent = create_test_agent("Agent", "Description", "Prompt")

    router.add_agent(agent)
    long_task = "process this data " * 500
    result = router.find_best_agent(long_task)

    assert result is not None


def test_special_characters_in_task():
    """Test task with special characters"""
    router = AgentRouter()
    agent = create_test_agent("Agent", "Description", "Prompt")

    router.add_agent(agent)
    result = router.find_best_agent("Task with @#$%^&*() symbols!")

    assert result is not None


def test_multiple_agents_same_description():
    """Test behavior when multiple agents have similar descriptions"""
    router = AgentRouter()
    agent1 = create_test_agent("Agent1", "Code expert", "Write code")
    agent2 = create_test_agent("Agent2", "Code expert", "Write code")

    router.add_agents([agent1, agent2])
    result = router.find_best_agent("Write a function")

    assert result in [agent1, agent2]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_add_find_update():
    """Test complete workflow of adding, finding, and updating agent"""
    router = AgentRouter()

    agent = create_test_agent(
        "DataAnalyst",
        "Expert in data analysis",
        "Analyze datasets and provide insights"
    )

    router.add_agent(agent)

    best_agent = router.find_best_agent("Analyze this dataset")
    assert best_agent == agent

    agent.short_memory.add("user", "analyze sales data")
    router.update_agent_history("DataAnalyst")

    assert len(router.agent_embeddings[0]) > 0


def test_concurrent_agent_additions():
    """Test adding agents sequentially maintains consistency"""
    router = AgentRouter()

    for i in range(5):
        agent = create_test_agent(
            f"Agent{i}",
            f"Description {i}",
            f"Prompt {i}"
        )
        router.add_agent(agent)

    assert len(router.agents) == 5
    assert len(router.agent_embeddings) == 5
    assert len(router.agent_metadata) == 5

    for i in range(5):
        assert router.agents[i].agent_name == f"Agent{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
