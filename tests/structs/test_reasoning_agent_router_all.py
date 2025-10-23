"""Testing all the parameters and methods of the reasoning agent router
- Parameters: description, model_name, system_prompt, max_loops, swarm_type, num_samples, output_types, num_knowledge_items, memory_capacity, eval, random_models_on, majority_voting_prompt, reasoning_model_name
- Methods: select_swarm(), run (task: str, img: Optional[List[str]] = None, **kwargs), batched_run (tasks: List[str], imgs: Optional[List[List[str]]] = None, **kwargs)
"""

import pytest
from swarms.agents import ReasoningAgentRouter

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_AGENT_NAME = "reasoning-agent"
DEFAULT_DESCRIPTION = (
    "A reasoning agent that can answer questions and help with tasks."
)
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."
DEFAULT_MAX_LOOPS = 1
DEFAULT_SWARM_TYPE = "self-consistency"
DEFAULT_NUM_SAMPLES = 3
DEFAULT_EVAL = False
DEFAULT_RANDOM_MODELS_ON = False
DEFAULT_MAJORITY_VOTING_PROMPT = None

# ============================================================================
# Helper Functions
# ============================================================================


def create_reasoning_agent_router(**kwargs):
    """Create a ReasoningAgentRouter with default or custom parameters."""
    params = {
        "agent_name": DEFAULT_AGENT_NAME,
        "description": DEFAULT_DESCRIPTION,
        "model_name": DEFAULT_MODEL_NAME,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_loops": DEFAULT_MAX_LOOPS,
        "swarm_type": DEFAULT_SWARM_TYPE,
        "num_samples": DEFAULT_NUM_SAMPLES,
        "eval": DEFAULT_EVAL,
        "random_models_on": DEFAULT_RANDOM_MODELS_ON,
        "majority_voting_prompt": DEFAULT_MAJORITY_VOTING_PROMPT,
        "verbose": False,  # Keep tests quiet
    }
    params.update(kwargs)
    return ReasoningAgentRouter(**params)

# ============================================================================
# Initialization Tests
# ============================================================================


def test_reasoning_agent_router_default_initialization():
    """Test ReasoningAgentRouter initialization with default parameters."""
    router = create_reasoning_agent_router()
    
    assert router.agent_name == DEFAULT_AGENT_NAME
    assert router.description == DEFAULT_DESCRIPTION
    assert router.model_name == DEFAULT_MODEL_NAME
    assert router.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert router.max_loops == DEFAULT_MAX_LOOPS
    assert router.swarm_type == DEFAULT_SWARM_TYPE
    assert router.num_samples == DEFAULT_NUM_SAMPLES
    assert router.eval == DEFAULT_EVAL
    assert router.random_models_on == DEFAULT_RANDOM_MODELS_ON


def test_reasoning_agent_router_custom_initialization():
    """Test ReasoningAgentRouter initialization with custom parameters."""
    router = create_reasoning_agent_router(
        agent_name="custom-reasoning-agent",
        description="Custom reasoning agent description",
        model_name="gpt-4o-mini",
        system_prompt="Custom system prompt",
        max_loops=3,
        swarm_type="mixture-of-agents",
        num_samples=5,
        eval=True,
        random_models_on=True,
    )
    
    assert router.agent_name == "custom-reasoning-agent"
    assert router.description == "Custom reasoning agent description"
    assert router.model_name == "gpt-4o-mini"
    assert router.system_prompt == "Custom system prompt"
    assert router.max_loops == 3
    assert router.swarm_type == "mixture-of-agents"
    assert router.num_samples == 5
    assert router.eval is True
    assert router.random_models_on is True

# ============================================================================
# Swarm Type Tests
# ============================================================================


def test_self_consistency_swarm():
    """Test ReasoningAgentRouter with self-consistency swarm type."""
    router = create_reasoning_agent_router(
        swarm_type="self-consistency",
        num_samples=3,
    )
    
    result = router.run("What is 2+2?")
    assert result is not None
    assert isinstance(result, str)


def test_mixture_of_agents_swarm():
    """Test ReasoningAgentRouter with mixture-of-agents swarm type."""
    router = create_reasoning_agent_router(
        swarm_type="mixture-of-agents",
        num_samples=3,
    )
    
    result = router.run("What is the capital of France?")
    assert result is not None
    assert isinstance(result, str)


def test_majority_voting_swarm():
    """Test ReasoningAgentRouter with majority-voting swarm type."""
    router = create_reasoning_agent_router(
        swarm_type="majority-voting",
        num_samples=3,
    )
    
    result = router.run("What is 5*6?")
    assert result is not None
    assert isinstance(result, str)

# ============================================================================
# Parameter Variation Tests
# ============================================================================


def test_different_model_names():
    """Test ReasoningAgentRouter with different model names."""
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    for model in models:
        router = create_reasoning_agent_router(model_name=model)
        assert router.model_name == model
        
        result = router.run("Simple test question")
        assert result is not None


def test_different_max_loops():
    """Test ReasoningAgentRouter with different max_loops values."""
    loop_values = [1, 2, 3]
    
    for loops in loop_values:
        router = create_reasoning_agent_router(max_loops=loops)
        assert router.max_loops == loops
        
        result = router.run("Test with different loops")
        assert result is not None


def test_different_num_samples():
    """Test ReasoningAgentRouter with different num_samples values."""
    sample_values = [1, 3, 5]
    
    for samples in sample_values:
        router = create_reasoning_agent_router(num_samples=samples)
        assert router.num_samples == samples
        
        result = router.run("Test with different samples")
        assert result is not None

# ============================================================================
# Method Tests
# ============================================================================


def test_select_swarm_method():
    """Test the select_swarm method."""
    router = create_reasoning_agent_router()
    
    # Test that select_swarm returns a swarm object
    swarm = router.select_swarm()
    assert swarm is not None


def test_run_method_basic():
    """Test the run method with basic task."""
    router = create_reasoning_agent_router()
    
    result = router.run("What is the square root of 16?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_run_method_with_image():
    """Test the run method with image parameter."""
    router = create_reasoning_agent_router()
    
    # Test with None image
    result = router.run("Describe this image", img=None)
    assert result is not None


def test_batched_run_method():
    """Test the batched_run method."""
    router = create_reasoning_agent_router()
    
    tasks = [
        "What is 1+1?",
        "What is 2+2?", 
        "What is 3+3?",
    ]
    
    results = router.batched_run(tasks)
    assert results is not None
    assert len(results) == len(tasks)
    assert all(result is not None for result in results)


def test_batched_run_with_images():
    """Test the batched_run method with images parameter."""
    router = create_reasoning_agent_router()
    
    tasks = ["Describe image 1", "Describe image 2"]
    images = [None, None]  # No actual images for testing
    
    results = router.batched_run(tasks, imgs=images)
    assert results is not None
    assert len(results) == len(tasks)

# ============================================================================
# Output Type Tests
# ============================================================================


def test_different_output_types():
    """Test ReasoningAgentRouter with different output types."""
    output_types = ["string", "dict", "list"]
    
    for output_type in output_types:
        router = create_reasoning_agent_router(output_type=output_type)
        
        result = router.run("Test output type")
        assert result is not None


def test_json_output_type():
    """Test ReasoningAgentRouter with JSON output type."""
    router = create_reasoning_agent_router(output_type="json")
    
    result = router.run("Return a JSON response")
    assert result is not None

# ============================================================================
# Advanced Configuration Tests
# ============================================================================


def test_with_evaluation_enabled():
    """Test ReasoningAgentRouter with evaluation enabled."""
    router = create_reasoning_agent_router(eval=True)
    
    assert router.eval is True
    
    result = router.run("Evaluate this response")
    assert result is not None


def test_with_random_models_enabled():
    """Test ReasoningAgentRouter with random models enabled."""
    router = create_reasoning_agent_router(random_models_on=True)
    
    assert router.random_models_on is True
    
    result = router.run("Use random models")
    assert result is not None


def test_with_custom_majority_voting_prompt():
    """Test ReasoningAgentRouter with custom majority voting prompt."""
    custom_prompt = "Choose the best answer from the following options:"
    
    router = create_reasoning_agent_router(
        swarm_type="majority-voting",
        majority_voting_prompt=custom_prompt,
    )
    
    assert router.majority_voting_prompt == custom_prompt
    
    result = router.run("Test majority voting")
    assert result is not None


def test_with_memory_capacity():
    """Test ReasoningAgentRouter with memory capacity configuration."""
    router = create_reasoning_agent_router(memory_capacity=1000)
    
    result = router.run("Test memory capacity")
    assert result is not None


def test_with_knowledge_items():
    """Test ReasoningAgentRouter with knowledge items configuration."""
    router = create_reasoning_agent_router(num_knowledge_items=5)
    
    result = router.run("Test knowledge items")
    assert result is not None

# ============================================================================
# Error Handling Tests
# ============================================================================


def test_empty_task():
    """Test ReasoningAgentRouter with empty task."""
    router = create_reasoning_agent_router()
    
    with pytest.raises((ValueError, TypeError)):
        router.run("")


def test_none_task():
    """Test ReasoningAgentRouter with None task."""
    router = create_reasoning_agent_router()
    
    with pytest.raises((ValueError, TypeError)):
        router.run(None)


def test_invalid_swarm_type():
    """Test ReasoningAgentRouter with invalid swarm type."""
    with pytest.raises((ValueError, TypeError)):
        create_reasoning_agent_router(swarm_type="invalid-swarm-type")


def test_invalid_num_samples():
    """Test ReasoningAgentRouter with invalid num_samples."""
    with pytest.raises((ValueError, TypeError)):
        create_reasoning_agent_router(num_samples=0)


def test_invalid_max_loops():
    """Test ReasoningAgentRouter with invalid max_loops."""
    with pytest.raises((ValueError, TypeError)):
        create_reasoning_agent_router(max_loops=0)

# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow():
    """Test complete ReasoningAgentRouter workflow."""
    router = create_reasoning_agent_router(
        agent_name="integration-test-router",
        description="Router for integration testing",
        swarm_type="self-consistency",
        num_samples=3,
        max_loops=1,
        eval=False,
    )
    
    # Test single task
    result = router.run("What is the meaning of life?")
    assert result is not None
    
    # Test batch tasks
    tasks = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
    ]
    
    batch_results = router.batched_run(tasks)
    assert len(batch_results) == 3
    assert all(result is not None for result in batch_results)


def test_different_configurations():
    """Test ReasoningAgentRouter with various configurations."""
    configurations = [
        {
            "swarm_type": "self-consistency",
            "num_samples": 3,
            "eval": False,
        },
        {
            "swarm_type": "mixture-of-agents", 
            "num_samples": 5,
            "eval": True,
        },
        {
            "swarm_type": "majority-voting",
            "num_samples": 4,
            "random_models_on": True,
        },
    ]
    
    for config in configurations:
        router = create_reasoning_agent_router(**config)
        
        result = router.run("Test configuration")
        assert result is not None
        
        # Verify configuration was applied
        for key, value in config.items():
            assert getattr(router, key) == value


def test_performance_with_multiple_samples():
    """Test ReasoningAgentRouter performance with multiple samples."""
    router = create_reasoning_agent_router(
        num_samples=5,
        swarm_type="self-consistency",
    )
    
    result = router.run("Complex reasoning task: solve 2x + 5 = 15")
    assert result is not None
    assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])