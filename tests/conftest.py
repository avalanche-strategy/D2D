import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from pathlib import Path
import shutil
from dotenv import load_dotenv
import logging
import torch
from sentence_transformers import SentenceTransformer


### fixture functions

@pytest.fixture
def test_case_files(tmp_path):
    def _copy_matching_case(test_case: str):
        """
        Fixture to copy all files and folders from the 'fixtures/' directory
        that have `test_case` as a substring in their filename or folder name
        into a temporary directory.

        Args:
            test_case (str): The name or substring to match fixture files or directories against.
        """
        fixtures_dir = Path(__file__).parent / "fixtures"

        # check whether the file or folder contains the case_name
        for item in fixtures_dir.iterdir():
            if test_case in item.name:
                dest = tmp_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

        return tmp_path
    return _copy_matching_case

@pytest.fixture
def test_case_files_with_extra_file(test_case_files):
    """
    Copies matching test case files into a temporary directory and creates an additional text file with specified content.

    Extends `test_case_files` fixture by adding a manually created text file
    to the temporary directory after copying the matching fixture files.

    Args:
        test_case (str): The name or substring to match fixture files or directories against.
        extra_filename (str, optional): Name of the extra file. Defaults to "dummy.txt".
        file_content (str, optional): Content to write into the created text file. Default value is empty string.

    Returns:
        pathlib.Path: Path to the temporary directory containing the copied fixture files and the extra file.
    """
    def _copy_and_add_file(test_case: str, extra_filename="dummy.txt", file_content=""):
        path = test_case_files(test_case)

        # add the new file with specified content
        extra_file = path / extra_filename
        extra_file.write_text(file_content)

        return path
    return _copy_and_add_file


@pytest.fixture()
def setup_bad_apikey(monkeypatch):
    """
    Fixture to override .env and set up an incorrect OPENAI_API_KEY. Tests using this fixture will use the incorrect OPENAI_API_KEY
    and fail with AuthenticationError

    Args:
        monkeypatch (Generator): monkeypatch generator
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'this-will-not-work')
    load_dotenv()

@pytest.fixture
def logger():
    """A fixture logger for test cases that require logging variable."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture
def embedding_model():
    """A fixture SentenceTransformer for test cases that require embedding."""
    sentence_model_name = "multi-qa-mpnet-base-dot-v1"
    return SentenceTransformer(sentence_model_name)

@pytest.fixture
def torch_device():
    """A fixture torch.device test cases that require device to be set"""
    torch_device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
        )
