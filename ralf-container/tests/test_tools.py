import os
import tempfile
import shutil
from ralf.tools import list_files, read_file, write_file, run_command

def test_file_tools():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test write_file
        file_path = os.path.join(tmpdir, "test.txt")
        result = write_file(file_path, "Hello World")
        assert "Successfully wrote" in result
        assert os.path.exists(file_path)

        # Test read_file
        content = read_file(file_path)
        assert content == "Hello World"

        # Test list_files
        files = list_files(tmpdir)
        assert "test.txt" in files

        # Test read_file non-existent
        result = read_file(os.path.join(tmpdir, "nonexistent.txt"))
        assert "Error reading file" in result

def test_run_command():
    result = run_command("echo Hello")
    assert "Hello" in result

    # Test a command that writes to stderr
    # On linux, ls of non-existent usually writes to stderr
    result = run_command("ls /nonexistent_directory_xyz")
    # Depending on shell and OS, might vary, but usually stderr is captured
    assert "stderr" in result
