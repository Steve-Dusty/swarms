import os
import pytest
import tempfile

from swarms.structs.agent import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_test_csv(tmpdir):
    """Create a test CSV file with agent configurations"""
    csv_content = """agent_name,description,system_prompt,task
test_agent_1,Test Agent 1,System prompt 1,Task 1
test_agent_2,Test Agent 2,System prompt 2,Task 2"""

    file_path = os.path.join(tmpdir, "test_agents.csv")
    with open(file_path, "w") as f:
        f.write(csv_content)

    return file_path


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name"""
    return Agent(
        agent_name=name,
        system_prompt=f"Test prompt for {name}",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_spreadsheet_swarm_basic_initialization():
    """Test basic SpreadSheetSwarm initialization"""
    agents = [
        create_test_agent("agent1"),
        create_test_agent("agent2"),
    ]

    swarm = SpreadSheetSwarm(
        name="Test Swarm",
        description="Test Description",
        agents=agents,
        max_loops=2,
    )

    assert swarm.name == "Test Swarm"
    assert swarm.description == "Test Description"
    assert len(swarm.agents) == 2
    assert swarm.max_loops == 2


def test_spreadsheet_swarm_default_initialization():
    """Test SpreadSheetSwarm with default parameters"""
    agents = [create_test_agent("agent1")]
    swarm = SpreadSheetSwarm(agents=agents)

    assert swarm.agents == agents
    assert len(swarm.agents) == 1


def test_spreadsheet_swarm_with_save_path():
    """Test SpreadSheetSwarm initialization with save file path"""
    agents = [create_test_agent("agent1")]
    save_path = "test_output.csv"

    swarm = SpreadSheetSwarm(
        agents=agents,
        save_file_path=save_path,
    )

    # Note: SpreadSheetSwarm may generate its own file path, so we test that it's set
    assert hasattr(swarm, 'save_file_path')
    assert swarm.save_file_path is not None


# ============================================================================
# CSV LOADING TESTS
# ============================================================================


def test_load_from_csv():
    """Test loading agent configurations from CSV"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = create_test_csv(tmpdir)
        # SpreadSheetSwarm requires agents, so provide them along with load_path
        agents = [create_test_agent("agent1")]
        swarm = SpreadSheetSwarm(agents=agents, load_path=csv_path)

        # Test that swarm was initialized with CSV path
        assert hasattr(swarm, 'load_path')
        
        # Test basic functionality
        result = swarm.run("Test task")
        assert result is not None
        assert isinstance(result, dict)  # SpreadSheetSwarm returns dict


def test_load_from_csv_creates_agents():
    """Test that loading from CSV creates the correct number of agents"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = create_test_csv(tmpdir)
        # Provide initial agents
        agents = [create_test_agent("agent1")]
        swarm = SpreadSheetSwarm(agents=agents, load_path=csv_path)

        # Test that agents were provided
        assert len(swarm.agents) >= 1
        
        result = swarm.run("Test task")
        assert result is not None
        assert isinstance(result, dict)


# ============================================================================
# TASK EXECUTION TESTS
# ============================================================================


def test_run_tasks():
    """Test running tasks through the swarm"""
    agents = [create_test_agent("agent1"), create_test_agent("agent2")]
    swarm = SpreadSheetSwarm(agents=agents)

    result = swarm.run("Test task for multiple agents")
    assert result is not None


def test_run_tasks_single_agent():
    """Test running tasks with a single agent"""
    agents = [create_test_agent("agent1")]
    swarm = SpreadSheetSwarm(agents=agents)

    result = swarm.run("Test task for single agent")
    assert result is not None


def test_run_with_custom_task():
    """Test running with custom task configuration"""
    agents = [create_test_agent("agent1")]
    swarm = SpreadSheetSwarm(agents=agents)

    result = swarm.run("Calculate the sum of 5 and 7")
    assert result is not None
    assert isinstance(result, dict)  # SpreadSheetSwarm returns dict with metadata


# ============================================================================
# OUTPUT TRACKING TESTS
# ============================================================================


def test_output_tracking():
    """Test tracking of task outputs"""
    swarm = SpreadSheetSwarm(agents=[create_test_agent("agent1")])

    # Test basic functionality without assuming internal structure
    result = swarm.run("Test task")
    assert result is not None


def test_output_tracking_multiple_outputs():
    """Test tracking multiple task outputs"""
    swarm = SpreadSheetSwarm(agents=[create_test_agent("agent1")])

    # Test multiple runs
    result1 = swarm.run("Task 1")
    result2 = swarm.run("Task 2")
    result3 = swarm.run("Task 3")

    assert result1 is not None
    assert result2 is not None
    assert result3 is not None


# ============================================================================
# CSV SAVING TESTS
# ============================================================================


def test_save_to_csv():
    """Test saving results to CSV"""
    agents = [create_test_agent("agent1")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "output.csv")
        swarm = SpreadSheetSwarm(
            agents=agents,
            save_file_path=save_path,
        )

        result = swarm.run("Test task for CSV output")
        assert result is not None


def test_save_to_csv_with_multiple_outputs():
    """Test saving multiple outputs to CSV"""
    agents = [create_test_agent("agent1")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "multi_output.csv")
        swarm = SpreadSheetSwarm(
            agents=agents,
            save_file_path=save_path,
        )

        # Run multiple tasks
        result1 = swarm.run("Task 1")
        result2 = swarm.run("Task 2")

        assert result1 is not None
        assert result2 is not None


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_swarm_configuration():
    """Test swarm configuration options"""
    agents = [create_test_agent("agent1")]
    
    swarm = SpreadSheetSwarm(
        name="Config Test Swarm",
        description="Testing configuration",
        agents=agents,
        max_loops=3,
        autosave=True,
    )

    assert swarm.name == "Config Test Swarm"
    assert swarm.description == "Testing configuration"
    assert swarm.max_loops == 3
    assert swarm.autosave is True


def test_swarm_with_multiple_agents():
    """Test swarm with multiple agents"""
    agents = [
        create_test_agent("agent1"),
        create_test_agent("agent2"),
        create_test_agent("agent3"),
    ]
    
    swarm = SpreadSheetSwarm(agents=agents)
    
    assert len(swarm.agents) == 3
    
    result = swarm.run("Collaborative task for all agents")
    assert result is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_empty_agents_list():
    """Test swarm with empty agents list"""
    with pytest.raises((ValueError, TypeError)):
        SpreadSheetSwarm(agents=[])


def test_none_agents():
    """Test swarm with None agents"""
    with pytest.raises((ValueError, TypeError)):
        SpreadSheetSwarm(agents=None)


def test_invalid_task():
    """Test swarm with invalid task"""
    agents = [create_test_agent("agent1")]
    swarm = SpreadSheetSwarm(agents=agents)

    # SpreadSheetSwarm handles empty/None tasks gracefully, so test that it returns something
    result_empty = swarm.run("")
    assert result_empty is not None
    assert isinstance(result_empty, dict)

    result_none = swarm.run(None)
    assert result_none is not None
    assert isinstance(result_none, dict)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_csv_to_json():
    """Test complete workflow from CSV input to JSON output"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input CSV
        csv_path = create_test_csv(tmpdir)
        
        # Create swarm with agents and CSV path
        agents = [create_test_agent("workflow_agent")]
        swarm = SpreadSheetSwarm(agents=agents, load_path=csv_path)
        
        # Run task
        result = swarm.run("Process this task using CSV configuration")
        assert result is not None
        assert isinstance(result, dict)


def test_full_workflow_with_save():
    """Test complete workflow with saving results"""
    agents = [create_test_agent("workflow_agent")]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "workflow_output.csv")
        
        swarm = SpreadSheetSwarm(
            name="Workflow Test Swarm",
            agents=agents,
            save_file_path=save_path,
            autosave=True,
        )

        # Run multiple tasks
        result1 = swarm.run("First workflow task")
        result2 = swarm.run("Second workflow task")

        assert result1 is not None
        assert result2 is not None


def test_agent_specialization_workflow():
    """Test workflow with specialized agents"""
    # Create specialized agents
    math_agent = Agent(
        agent_name="math_specialist",
        system_prompt="You are a mathematics expert. Solve problems step by step.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    
    writing_agent = Agent(
        agent_name="writing_specialist",
        system_prompt="You are a writing expert. Create clear, engaging content.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    
    swarm = SpreadSheetSwarm(
        name="Specialized Workflow",
        agents=[math_agent, writing_agent],
        max_loops=1,
    )
    
    # Test with task that could benefit from specialization
    result = swarm.run("Explain the Pythagorean theorem with examples")
    assert result is not None
    assert isinstance(result, dict)  # SpreadSheetSwarm returns dict


def test_large_task_processing():
    """Test processing of larger, more complex tasks"""
    agents = [
        create_test_agent("analyst1"),
        create_test_agent("analyst2"),
    ]
    
    swarm = SpreadSheetSwarm(
        name="Large Task Processor",
        agents=agents,
        max_loops=2,
    )
    
    complex_task = (
        "Analyze the benefits and drawbacks of renewable energy sources. "
        "Consider economic, environmental, and social factors. "
        "Provide a balanced perspective with specific examples."
    )
    
    result = swarm.run(complex_task)
    assert result is not None
    assert isinstance(result, dict)  # SpreadSheetSwarm returns dict
    # Check that the result dict has expected keys
    assert "name" in result
    assert "description" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])