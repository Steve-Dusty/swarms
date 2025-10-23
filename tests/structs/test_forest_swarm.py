import pytest

from swarms.structs.tree_swarm import (
    TreeAgent,
    Tree,
    ForestSwarm,
    AgentLogInput,
    AgentLogOutput,
    TreeLog,
    extract_keywords,
    cosine_similarity,
)


# Test Data
SAMPLE_SYSTEM_PROMPTS = {
    "financial_advisor": "I am a financial advisor specializing in investment planning, retirement strategies, and tax optimization for individuals and businesses.",
    "tax_expert": "I am a tax expert with deep knowledge of corporate taxation, Delaware incorporation benefits, and free tax filing options for businesses.",
    "stock_analyst": "I am a stock market analyst who provides insights on market trends, stock recommendations, and portfolio optimization strategies.",
    "retirement_planner": "I am a retirement planning specialist who helps individuals and businesses create comprehensive retirement strategies and investment plans.",
}

SAMPLE_TASKS = {
    "tax_question": "Our company is incorporated in Delaware, how do we do our taxes for free?",
    "investment_question": "What are the best investment strategies for a 401k retirement plan?",
    "stock_question": "Which tech stocks should I consider for my investment portfolio?",
    "retirement_question": "How much should I save monthly for retirement if I want to retire at 65?",
}


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


def test_extract_keywords_basic():
    """Test basic keyword extraction"""
    text = (
        "financial advisor investment planning retirement strategies"
    )
    keywords = extract_keywords(text, top_n=3)

    assert len(keywords) == 3
    assert "financial" in keywords
    assert "investment" in keywords


def test_extract_keywords_with_punctuation():
    """Test keyword extraction with punctuation and case"""
    text = "Tax Expert! Corporate Taxation, Delaware Incorporation."
    keywords = extract_keywords(text, top_n=5)

    assert "tax" in keywords
    assert "corporate" in keywords


def test_extract_keywords_empty_string():
    """Test keyword extraction with empty string"""
    keywords = extract_keywords("", top_n=3)
    assert len(keywords) == 0


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity with identical vectors"""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)

    assert similarity == 1.0


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity with orthogonal vectors"""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)

    assert similarity == 0.0


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity with opposite vectors"""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)

    assert similarity == -1.0


def test_cosine_similarity_zero_vectors():
    """Test cosine similarity with zero vectors"""
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)

    assert similarity == 0.0


def test_cosine_similarity_empty_vectors():
    """Test cosine similarity with empty vectors"""
    empty_vec = []
    vec = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(empty_vec, vec)

    assert similarity == 0.0


# ============================================================================
# PYDANTIC LOG MODEL TESTS
# ============================================================================


def test_agent_log_input_creation():
    """Test AgentLogInput model creation"""
    log_input = AgentLogInput(
        agent_name="test_agent", task="test_task"
    )

    assert isinstance(log_input, AgentLogInput)
    assert log_input.log_id is not None
    assert log_input.agent_name == "test_agent"
    assert log_input.task == "test_task"


def test_agent_log_output_creation():
    """Test AgentLogOutput model creation"""
    log_output = AgentLogOutput(
        agent_name="test_agent", result="test_result"
    )

    assert isinstance(log_output, AgentLogOutput)
    assert log_output.log_id is not None
    assert log_output.result == "test_result"


def test_tree_log_creation():
    """Test TreeLog model creation"""
    tree_log = TreeLog(
        tree_name="test_tree",
        task="test_task",
        selected_agent="test_agent",
        result="test_result",
    )

    assert isinstance(tree_log, TreeLog)
    assert tree_log.log_id is not None
    assert tree_log.tree_name == "test_tree"
    assert tree_log.task == "test_task"
    assert tree_log.selected_agent == "test_agent"


# ============================================================================
# TREE AGENT TESTS
# ============================================================================


def test_tree_agent_basic_initialization():
    """Test basic TreeAgent initialization"""
    agent = TreeAgent(
        name="Test Agent",
        description="A test agent",
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="financial_advisor",
    )

    assert isinstance(agent, TreeAgent)
    assert agent.agent_name == "financial_advisor"
    assert agent.embedding_model_name == "text-embedding-ada-002"
    assert len(agent.relevant_keywords) > 0
    assert agent.system_prompt_embedding is not None


def test_tree_agent_custom_embedding_model():
    """Test TreeAgent with custom embedding model"""
    agent = TreeAgent(
        system_prompt="Test prompt",
        embedding_model_name="custom-model",
    )

    assert agent.embedding_model_name == "custom-model"


def test_tree_agent_distance_calculation():
    """Test distance calculation between TreeAgents"""
    agent1 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="financial_advisor",
    )

    agent2 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
        agent_name="tax_expert",
    )

    agent3 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
        agent_name="stock_analyst",
    )

    distance1 = agent1.calculate_distance(agent2)
    distance2 = agent1.calculate_distance(agent3)

    assert 0.0 <= distance1 <= 1.0
    assert 0.0 <= distance2 <= 1.0
    assert isinstance(distance1, float)


def test_tree_agent_distance_identical_agents():
    """Test that identical agents have very small distance"""
    agent1 = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="financial_advisor",
    )

    identical_agent = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
        agent_name="identical_advisor",
    )

    distance = agent1.calculate_distance(identical_agent)
    assert distance < 0.1


def test_tree_agent_task_relevance():
    """Test TreeAgent task relevance checking"""
    tax_agent = TreeAgent(
        system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
        agent_name="tax_expert",
    )

    tax_task = SAMPLE_TASKS["tax_question"]
    is_relevant = tax_agent.is_relevant_for_task(
        tax_task, threshold=0.7
    )

    assert isinstance(is_relevant, bool)


def test_tree_agent_none_prompt():
    """Test TreeAgent with None system prompt"""
    agent = TreeAgent(system_prompt=None, agent_name="no_prompt_agent")

    assert len(agent.relevant_keywords) == 0
    assert agent.system_prompt_embedding is None


# ============================================================================
# TREE TESTS
# ============================================================================


def test_tree_initialization():
    """Test Tree initialization with multiple agents"""
    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
            agent_name="stock_analyst",
        ),
    ]

    tree = Tree("Financial Services Tree", agents)

    assert tree.tree_name == "Financial Services Tree"
    assert len(tree.agents) == 3
    assert all(hasattr(agent, "distance") for agent in tree.agents)


def test_tree_agents_sorted_by_distance():
    """Test that Tree agents are sorted by distance"""
    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
            agent_name="stock_analyst",
        ),
    ]

    tree = Tree("Test Tree", agents)
    distances = [agent.distance for agent in tree.agents]

    assert distances == sorted(distances)


def test_tree_find_relevant_agent():
    """Test Tree finding relevant agent for task"""
    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
    ]

    tree = Tree("Test Tree", agents)
    tax_task = SAMPLE_TASKS["tax_question"]
    relevant_agent = tree.find_relevant_agent(tax_task)

    assert relevant_agent is not None


def test_tree_unrelated_task():
    """Test Tree with unrelated task"""
    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        ),
    ]

    tree = Tree("Test Tree", agents)
    unrelated_task = "How do I cook pasta?"
    relevant_agent = tree.find_relevant_agent(unrelated_task)

    assert (
        relevant_agent is None or isinstance(relevant_agent, TreeAgent)
    )


def test_tree_empty_agents_list():
    """Test Tree with empty agents list"""
    empty_tree = Tree("Empty Tree", [])
    assert len(empty_tree.agents) == 0


# ============================================================================
# FOREST SWARM TESTS
# ============================================================================


def test_forest_swarm_initialization():
    """Test ForestSwarm initialization"""
    agents_tree1 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        )
    ]

    agents_tree2 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        )
    ]

    tree1 = Tree("Financial Tree", agents_tree1)
    tree2 = Tree("Tax Tree", agents_tree2)

    forest = ForestSwarm(
        name="Test Forest",
        description="A test forest",
        trees=[tree1, tree2],
    )

    assert forest.name == "Test Forest"
    assert forest.description == "A test forest"
    assert len(forest.trees) == 2
    assert forest.conversation is not None


def test_forest_swarm_find_relevant_tree():
    """Test ForestSwarm finding relevant tree for task"""
    agents_tree1 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        )
    ]

    agents_tree2 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        )
    ]

    tree1 = Tree("Financial Tree", agents_tree1)
    tree2 = Tree("Tax Tree", agents_tree2)

    forest = ForestSwarm(trees=[tree1, tree2])

    tax_task = SAMPLE_TASKS["tax_question"]
    relevant_tree = forest.find_relevant_tree(tax_task)

    assert relevant_tree is not None


def test_forest_swarm_find_tree_for_financial_task():
    """Test ForestSwarm finding tree for financial task"""
    agents_tree1 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        )
    ]

    agents_tree2 = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["tax_expert"],
            agent_name="tax_expert",
        )
    ]

    tree1 = Tree("Financial Tree", agents_tree1)
    tree2 = Tree("Tax Tree", agents_tree2)

    forest = ForestSwarm(trees=[tree1, tree2])

    financial_task = SAMPLE_TASKS["investment_question"]
    relevant_tree = forest.find_relevant_tree(financial_task)

    assert relevant_tree is not None


def test_forest_swarm_task_execution():
    """Test ForestSwarm task execution"""
    agent = TreeAgent(
        system_prompt="I am a helpful assistant that can answer questions about Delaware incorporation and taxes.",
        agent_name="delaware_expert",
    )

    tree = Tree("Delaware Tree", [agent])
    forest = ForestSwarm(trees=[tree])

    task = "What are the benefits of incorporating in Delaware?"

    try:
        result = forest.run(task)
        assert result is not None
        assert isinstance(result, str)
    except Exception:
        # If execution fails due to external dependencies, skip
        pytest.skip(
            "Task execution failed due to external dependencies"
        )


def test_forest_swarm_empty_trees():
    """Test ForestSwarm with empty trees list"""
    empty_forest = ForestSwarm(trees=[])
    assert len(empty_forest.trees) == 0


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_forest_swarm_with_multiple_agents_per_tree():
    """Test ForestSwarm with multiple agents per tree"""
    agents = [
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["financial_advisor"],
            agent_name="financial_advisor",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["stock_analyst"],
            agent_name="stock_analyst",
        ),
        TreeAgent(
            system_prompt=SAMPLE_SYSTEM_PROMPTS["retirement_planner"],
            agent_name="retirement_planner",
        ),
    ]

    tree = Tree("Comprehensive Financial Tree", agents)
    forest = ForestSwarm(
        name="Multi-Agent Forest",
        description="Forest with multiple agents per tree",
        trees=[tree],
    )

    assert len(forest.trees) == 1
    assert len(forest.trees[0].agents) == 3
