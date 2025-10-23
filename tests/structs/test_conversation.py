import json
import shutil
import pytest
from pathlib import Path

from swarms.structs.conversation import Conversation


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path("temp_test_conversations")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    return temp_dir


def cleanup_temp_files(patterns=None):
    """Clean up temporary files created during tests."""
    import os
    import glob

    if patterns is None:
        patterns = ["conversation_*.json", "conversation_*.yaml", "temp_test_*"]

    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            except:
                pass


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_initialization_default():
    """Test default Conversation initialization"""
    conv = Conversation()

    assert conv.conversation_history == []
    assert conv.time_enabled is True
    assert conv.autosave is False
    assert conv.message_id_on is False


def test_initialization_with_name():
    """Test initialization with custom name"""
    conv = Conversation(name="test_conversation")

    assert conv.name == "test_conversation"


def test_initialization_with_time_enabled():
    """Test initialization with time enabled"""
    conv = Conversation(time_enabled=True)

    assert conv.time_enabled is True


def test_initialization_with_message_id():
    """Test initialization with message_id enabled"""
    conv = Conversation(message_id_on=True)

    assert conv.message_id_on is True


# ============================================================================
# MESSAGE OPERATIONS TESTS
# ============================================================================


def test_add_message_basic():
    """Test adding a basic message"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Hello, world!")

    assert len(conv.conversation_history) == 1
    assert conv.conversation_history[0]["role"] == "user"
    assert conv.conversation_history[0]["content"] == "Hello, world!"


def test_add_message_with_timestamp():
    """Test that timestamp is added when time_enabled=True"""
    conv = Conversation(time_enabled=True)
    conv.add("user", "Test message")

    assert "timestamp" in conv.conversation_history[0]
    assert isinstance(conv.conversation_history[0]["timestamp"], str)


def test_add_message_with_message_id():
    """Test that message_id is added when message_id_on=True"""
    conv = Conversation(message_id_on=True, time_enabled=False)
    conv.add("user", "Test message")

    assert "message_id" in conv.conversation_history[0]


def test_add_multiple_messages():
    """Test adding multiple messages"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "First message")
    conv.add("assistant", "Second message")
    conv.add("user", "Third message")

    assert len(conv.conversation_history) == 3
    assert conv.conversation_history[0]["content"] == "First message"
    assert conv.conversation_history[1]["content"] == "Second message"
    assert conv.conversation_history[2]["content"] == "Third message"


def test_update_message():
    """Test updating an existing message"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Original message")
    conv.update(0, "user", "Updated message")

    assert len(conv.conversation_history) == 1
    assert conv.conversation_history[0]["content"] == "Updated message"


def test_update_message_invalid_index():
    """Test that updating invalid index doesn't crash"""
    conv = Conversation()
    conv.add("user", "Test")

    original_length = len(conv.conversation_history)
    conv.update(10, "user", "Should not be added")

    assert len(conv.conversation_history) == original_length


def test_delete_message():
    """Test deleting a message"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Message 1")
    conv.add("user", "Message 2")
    conv.add("user", "Message 3")

    conv.delete(1)

    assert len(conv.conversation_history) == 2
    assert conv.conversation_history[0]["content"] == "Message 1"
    assert conv.conversation_history[1]["content"] == "Message 3"


def test_delete_message_invalid_index():
    """Test that deleting invalid index doesn't crash"""
    conv = Conversation()
    conv.add("user", "Test")

    original_length = len(conv.conversation_history)
    conv.delete(10)

    assert len(conv.conversation_history) == original_length


def test_clear_conversation():
    """Test clearing all messages"""
    conv = Conversation()
    conv.add("user", "Message 1")
    conv.add("user", "Message 2")

    conv.clear()

    assert len(conv.conversation_history) == 0


# ============================================================================
# STRING FORMATTING TESTS
# ============================================================================


def test_return_history_as_string_basic():
    """Test basic string formatting of conversation history"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Hello")
    conv.add("assistant", "Hi there")

    result = conv.return_history_as_string()

    assert "user: Hello" in result
    assert "assistant: Hi there" in result


def test_return_history_as_string_empty():
    """Test string formatting with empty conversation"""
    conv = Conversation()

    result = conv.return_history_as_string()

    assert result == ""


def test_get_str_shorthand():
    """Test get_str() as shorthand for return_history_as_string()"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Test")

    assert conv.get_str() == conv.return_history_as_string()


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


def test_save_as_json():
    """Test saving conversation as JSON"""
    try:
        conv = Conversation(time_enabled=False)
        conv.add("user", "Test message")

        filename = "test_conversation.json"
        conv.save_as_json(filename)

        assert Path(filename).exists()

        with open(filename, 'r') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "Test message"

    finally:
        cleanup_temp_files(["test_conversation.json"])


def test_save_as_yaml():
    """Test saving conversation as YAML"""
    try:
        conv = Conversation(time_enabled=False)
        conv.add("user", "Test message")

        filename = "test_conversation.yaml"
        conv.save_as_yaml(filename)

        assert Path(filename).exists()

    finally:
        cleanup_temp_files(["test_conversation.yaml"])


def test_load_from_json():
    """Test loading conversation from JSON file"""
    try:
        filename = "test_load.json"

        conv1 = Conversation(time_enabled=False)
        conv1.add("user", "Message 1")
        conv1.add("assistant", "Message 2")
        conv1.save_as_json(filename)

        conv2 = Conversation()
        conv2.load(filename)

        assert len(conv2.conversation_history) == 2
        assert conv2.conversation_history[0]["content"] == "Message 1"
        assert conv2.conversation_history[1]["content"] == "Message 2"

    finally:
        cleanup_temp_files(["test_load.json"])


# ============================================================================
# QUERY AND SEARCH TESTS
# ============================================================================


def test_query_by_role():
    """Test querying messages by role"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "User message 1")
    conv.add("assistant", "Assistant message")
    conv.add("user", "User message 2")

    user_messages = conv.query(role="user")

    assert len(user_messages) == 2
    assert all(msg["role"] == "user" for msg in user_messages)


def test_query_by_keyword():
    """Test querying messages by keyword in content"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Tell me about Python")
    conv.add("assistant", "Python is a programming language")
    conv.add("user", "What about JavaScript?")

    python_messages = conv.query(keyword="Python")

    assert len(python_messages) >= 1
    assert any("Python" in msg["content"] for msg in python_messages)


def test_search_keyword():
    """Test searching for keyword in messages"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Hello world")
    conv.add("user", "Test message")
    conv.add("user", "Hello again")

    results = conv.search("Hello")

    assert len(results) == 2


# ============================================================================
# EXPORT TESTS
# ============================================================================


def test_export_to_dict():
    """Test exporting conversation to dictionary"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "Test message")

    result = conv.export()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Test message"


def test_export_to_txt():
    """Test exporting conversation to text file"""
    try:
        conv = Conversation(time_enabled=False)
        conv.add("user", "Message 1")
        conv.add("assistant", "Message 2")

        filename = "test_export.txt"
        conv.export_to_txt(filename)

        assert Path(filename).exists()

        with open(filename, 'r') as f:
            content = f.read()
            assert "Message 1" in content
            assert "Message 2" in content

    finally:
        cleanup_temp_files(["test_export.txt"])


# ============================================================================
# RULES AND CONTEXT TESTS
# ============================================================================


def test_conversation_with_rules():
    """Test conversation initialization with rules"""
    rules = "Be helpful and concise"
    conv = Conversation(rules=rules, time_enabled=False)

    history_str = conv.return_history_as_string()

    assert rules in history_str


# ============================================================================
# EDGE CASES
# ============================================================================


def test_add_empty_content():
    """Test adding message with empty content"""
    conv = Conversation()
    conv.add("user", "")

    assert len(conv.conversation_history) == 1
    assert conv.conversation_history[0]["content"] == ""


def test_add_very_long_message():
    """Test adding very long message"""
    conv = Conversation(time_enabled=False)
    long_message = "x" * 100000
    conv.add("user", long_message)

    assert len(conv.conversation_history) == 1
    assert len(conv.conversation_history[0]["content"]) == 100000


def test_special_characters_in_message():
    """Test message with special characters"""
    conv = Conversation(time_enabled=False)
    special_msg = "Hello @#$%^&*() <html> {json} [array]"
    conv.add("user", special_msg)

    assert conv.conversation_history[0]["content"] == special_msg


def test_unicode_in_message():
    """Test message with Unicode characters"""
    conv = Conversation(time_enabled=False)
    unicode_msg = "Hello 世界 🌍 مرحبا"
    conv.add("user", unicode_msg)

    assert conv.conversation_history[0]["content"] == unicode_msg


# ============================================================================
# COUNT AND STATISTICS TESTS
# ============================================================================


def test_count_messages():
    """Test counting messages in conversation"""
    conv = Conversation()
    conv.add("user", "Message 1")
    conv.add("assistant", "Message 2")
    conv.add("user", "Message 3")

    count = conv.count_messages()

    assert count == 3


def test_count_messages_by_role():
    """Test counting messages by specific role"""
    conv = Conversation(time_enabled=False)
    conv.add("user", "User 1")
    conv.add("assistant", "Assistant")
    conv.add("user", "User 2")

    user_count = conv.count_messages_by_role("user")
    assistant_count = conv.count_messages_by_role("assistant")

    assert user_count == 2
    assert assistant_count == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow():
    """Test complete workflow: create, add, save, load"""
    try:
        filename = "test_workflow.json"

        conv1 = Conversation(name="workflow_test", time_enabled=False)
        conv1.add("user", "Start conversation")
        conv1.add("assistant", "Hello!")
        conv1.add("user", "How are you?")
        conv1.save_as_json(filename)

        conv2 = Conversation()
        conv2.load(filename)

        assert len(conv2.conversation_history) == 3
        assert conv2.conversation_history[0]["content"] == "Start conversation"

        result_str = conv2.return_history_as_string()
        assert "Start conversation" in result_str
        assert "Hello!" in result_str

    finally:
        cleanup_temp_files(["test_workflow.json"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
