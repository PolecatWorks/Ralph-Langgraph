import os
from ralph.agent import write_file

def test_write_file_root():
    """Test writing a file to the current directory (root)."""
    filename = "test_root_file.txt"
    content = "Hello, World!"

    # Ensure cleanup
    if os.path.exists(filename):
        os.remove(filename)

    try:
        result = write_file.invoke({"path": filename, "content": content})
        assert result == f"Successfully wrote to {filename}"
        assert os.path.exists(filename)
        with open(filename, "r") as f:
            assert f.read() == content
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_write_file_nested():
    """Test writing a file to a nested directory."""
    filename = "nested/dir/test_nested_file.txt"
    content = "Nested Hello"

    # Ensure cleanup
    if os.path.exists(filename):
        os.remove(filename)
        # Clean up dirs if empty? better to leave them or use tempdir via pytest

    try:
        result = write_file.invoke({"path": filename, "content": content})
        assert result == f"Successfully wrote to {filename}"
        assert os.path.exists(filename)
        with open(filename, "r") as f:
            assert f.read() == content
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            try:
                os.removedirs(os.path.dirname(filename))
            except OSError:
                pass
