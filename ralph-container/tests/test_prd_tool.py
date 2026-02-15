import os
import json
import shutil
import pytest
from unittest.mock import MagicMock
from ralph.agent import update_prd, _get_workdir

# Mock RunnableConfig
@pytest.fixture
def mock_config(tmpdir):
    config = MagicMock()
    config.get.return_value = {"workdir": str(tmpdir)}
    return config

def test_update_prd_creates_new_file(mock_config, tmpdir):
    """Test that update_prd creates a new prd.json if it doesn't exist."""
    story_title = "Test Story 1"
    result = update_prd.func(story_title, mock_config)

    assert "Successfully added story" in result

    prd_path = os.path.join(str(tmpdir), "prd.json")
    assert os.path.exists(prd_path)

    with open(prd_path, "r") as f:
        data = json.load(f)

    assert data["branchName"] == "main"
    assert len(data["userStories"]) == 1
    assert data["userStories"][0]["storyTitle"] == story_title
    assert data["userStories"][0]["passes"] is False

def test_update_prd_appends_to_existing_file(mock_config, tmpdir):
    """Test that update_prd appends to an existing prd.json and maintains formatting."""
    prd_path = os.path.join(str(tmpdir), "prd.json")

    # Create initial file with one story
    initial_data = {
        "branchName": "main",
        "userStories": [
            {"storyId": "1", "storyTitle": "Existing Story", "passes": True}
        ]
    }
    with open(prd_path, "w") as f:
        json.dump(initial_data, f)

    # Add second story
    story_title = "Test Story 2"
    result = update_prd.func(story_title, mock_config, notes="Some notes")

    assert "Successfully added story" in result

    with open(prd_path, "r") as f:
        content = f.read()
        data = json.loads(content)

    assert len(data["userStories"]) == 2
    assert data["userStories"][1]["storyTitle"] == story_title
    assert data["userStories"][1]["notes"] == "Some notes"

    # Check formatting (heuristic: check for indentation)
    # json.dump(indent=2) usually results in keys being on new lines with indentation
    assert '  "userStories": [' in content
    assert '    {' in content
    assert '      "storyId":' in content

def test_update_prd_handles_custom_id(mock_config, tmpdir):
    """Test that update_prd uses the provided story_id."""
    story_title = "Custom ID Story"
    story_id = "CUSTOM-123"
    update_prd.func(story_title, mock_config, story_id=story_id)

    prd_path = os.path.join(str(tmpdir), "prd.json")
    with open(prd_path, "r") as f:
        data = json.load(f)

    assert data["userStories"][0]["storyId"] == story_id
