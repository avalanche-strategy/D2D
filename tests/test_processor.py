import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from d2d import D2DProcessor
from pathlib import Path
import shutil
import pandas as pd
import json
import uuid
import requests
from unittest.mock import patch, mock_open
from litellm.exceptions import AuthenticationError, APIError, Timeout


### Test the entire pipeline as a blackbox

def test_csv_valid_interview_default(subtests, test_case_files):
    """
    Tests that loading a simple valid interview set (guidelines + transcripts) with default settings.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """

    test_case = "000"
    temp_folder = test_case_files(test_case)
    transcripts_path = temp_folder / f"interview_{test_case}"
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()

    # create output
    output_folder = temp_folder / "results"
    output_folder.mkdir()

    # Step 1: Initialize the processor to use all defaults
    processor = D2DProcessor()

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        transcripts_dir=str(transcripts_path),
        guidelines_path=str(guidelines_file),
        interview_name=f"interview_{test_case}",
        output_dir=output_folder,
        disable_logging_to_console=True
    )

    # check that the output folder exists - should be True
    with subtests.test("Output folder exists"):
        assert output_folder.exists()

    # confirm that the folder is not empty
    with subtests.test("Output folder not empty"):
        if not any(output_folder.iterdir()):
            raise AssertionError(f"{output_folder} is empty.")

    # dump content of all files in the output folder
    for filename in os.listdir(str(output_folder)):
        file_path = os.path.join(str(output_folder), filename)
        print(f"--- Contents of {filename} ---")
        with open(file_path, 'r') as file:
            print(file.read())
        print("\n\n")  # separator

        # test the content in the CSV file produced in output
        if (filename.endswith(".csv")):
            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you feeling at work?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"] == "fe4b129c-a507"]
            with subtests.test("Output CSV row has expected values"):
                assert test_row["How are you feeling at work?"].str.contains("overwhelmed", case=False).all()
                assert (test_row["What plans do you have for summer?"] == "[No relevant response found]").all()

        # test the content in the JSON file produced in output
        if (filename.endswith(".json")):
            with open(file_path) as out_file:
                data = json.load(out_file)
                with subtests.test("Output contains one interview"):
                    assert len(data) == 1  # this file only has 1 interview
                # there should be 2 responses in the file
                with subtests.test("Output for interview has 2 responses"):
                    assert len(data[0]["responses"]) == 2
                # positive response
                pos_resp = (
                    next((obj for obj in data[0]["responses"]
                          if obj["guide_question"] == "How are you feeling at work?"), None)
                )
                with subtests.test("Positive response column exists"):
                    assert pos_resp
                with subtests.test("Positive response references correct line"):
                    assert pos_resp["relevant_lines"] == [[1, 3], [5, 7]]
                with subtests.test("Positive response extracts correct references"):
                    assert pos_resp["extracted_line_references"] == [7]

                # negative response
                neg_resp = (
                    next((obj for obj in data[0]["responses"]
                          if obj["guide_question"] == "What plans do you have for summer?"), None)
                )
                with subtests.test("Negative response column exists"):
                    assert neg_resp
                with subtests.test("Negative response has 0 references"):
                    assert neg_resp["relevant_lines"] == [[1, 3], [5, 7]]
                with subtests.test("Negative response does not extract references"):
                    assert neg_resp["extracted_line_references"] is None


def test_csv_valid_interview_plus_empty(subtests, test_case_files_with_extra_file):
    """
    Tests that loading a simple valid interview set (guidelines + transcripts), with an additional
        empty transcript works as expected with the default settings.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """

    test_case = "000"
    extra_file_name = f"{str(uuid.uuid4())}.txt"
    temp_folder = test_case_files_with_extra_file(test_case,
                                                  extra_filename=extra_file_name,
                                                  file_content="This file is empty.")
    transcripts_path = temp_folder / f"interview_{test_case}"
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()

    # create output
    output_folder = temp_folder / "results"
    output_folder.mkdir()

    # Step 1: Initialize the processor use all defaults
    processor = D2DProcessor(
        sampling_method=D2DProcessor.SamplingMethod.TOP_P
    )

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        transcripts_dir=str(transcripts_path),
        guidelines_path=str(guidelines_file),
        interview_name=f"interview_{test_case}",
        output_dir=output_folder,
        disable_logging_to_console=True
    )

    # check that the output folder exists - should be True
    with subtests.test("Output folder exists"):
        assert output_folder.exists()

    # confirm that the folder is not empty
    with subtests.test("Output folder not empty"):
        if not any(output_folder.iterdir()):
            raise AssertionError(f"{output_folder} is empty.")

    # dump content of all files in the output folder for debug
    # enable output of print statement with -s flag in pytest (`pytest -s`)
    for filename in os.listdir(str(output_folder)):
        file_path = os.path.join(str(output_folder), filename)
        print(f"--- Contents of {filename} ---")
        with open(file_path, 'r') as file:
            print(file.read())
        print("\n\n")  # separator

        # test the content in the CSV file produced in output
        if (filename.endswith(".csv")):
            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you feeling at work?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"] == extra_file_name]
            with subtests.test("Output CSV row has expected values"):
                assert (test_row["How are you feeling at work?"] == "[No relevant response found]").all()
                assert (test_row["What plans do you have for summer?"] == "[No relevant response found]").all()


def test_csv_valid_interview_top_p(subtests, test_case_files):
    """
    Tests that loading a simple valid interview set (guidelines + transcripts) works when matching set to top-p.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """

    test_case = "000"
    temp_folder = test_case_files(test_case)
    transcripts_path = temp_folder / f"interview_{test_case}"
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()

    # create output
    output_folder = temp_folder / "results"
    output_folder.mkdir()

    # Step 1: Initialize the processor to use all defaults except matching method
    processor = D2DProcessor(sampling_method=D2DProcessor.SamplingMethod.TOP_P)

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        transcripts_dir=str(transcripts_path),
        guidelines_path=str(guidelines_file),
        interview_name=f"interview_{test_case}",
        output_dir=output_folder,
        disable_logging_to_console=True
    )

    # check that the output folder exists - should be True
    with subtests.test("Output folder exists"):
        assert output_folder.exists()

    # confirm that the folder is not empty
    with subtests.test("Output folder not empty"):
        if not any(output_folder.iterdir()):
            raise AssertionError(f"{output_folder} is empty.")

    # dump content of all files in the output folder
    for filename in os.listdir(str(output_folder)):
        file_path = os.path.join(str(output_folder), filename)
        print(f"--- Contents of {filename} ---")
        with open(file_path, 'r') as file:
            print(file.read())
        print("\n\n")  # separator

        # test the content in the CSV file produced in output
        if (filename.endswith(".csv")):
            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you feeling at work?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"] == "fe4b129c-a507"]
            with subtests.test("Output CSV row has expected values"):
                assert test_row["How are you feeling at work?"].str.contains("overwhelmed", case=False).all()
                assert (test_row["What plans do you have for summer?"] == "[No relevant response found]").all()

        # test the content in the JSON file produced in output
        if (filename.endswith(".json")):
            with open(file_path) as out_file:
                data = json.load(out_file)
                with subtests.test("Output contains one interview"):
                    assert len(data) == 1  # this file only has 1 interview
                # there should be 2 responses in the file
                with subtests.test("Output for interview has 2 responses"):
                    assert len(data[0]["responses"]) == 2
                # positive response
                pos_resp = (
                    next((obj for obj in data[0]["responses"]
                          if obj["guide_question"] == "How are you feeling at work?"), None)
                )
                with subtests.test("Positive response column exists"):
                    assert pos_resp
                with subtests.test("Positive response references correct line"):
                    # only this QA pair should be retrieved by top-p
                    assert pos_resp["relevant_lines"] == [[5, 7]]
                with subtests.test("Positive response extracts correct references"):
                    assert pos_resp["extracted_line_references"] == [7]

                # negative response
                neg_resp = (
                    next((obj for obj in data[0]["responses"]
                          if obj["guide_question"] == "What plans do you have for summer?"), None)
                )
                with subtests.test("Negative response column exists"):
                    assert neg_resp
                with subtests.test("Negative response has 0 references"):
                    # there should be no valid relevant lines  to the second question
                    assert neg_resp["relevant_lines"] == []
                with subtests.test("Negative response does not extract references"):
                    assert neg_resp["extracted_line_references"] is None


### Unit tests for _check_internet_connection and _test_llm_connection

@pytest.fixture
def processor():
    """Fixture to initialize D2DProcessor with default settings."""
    return D2DProcessor(llm_model="gpt-4o-mini")


def test_check_internet_connection_success(processor, mocker):
    """Test _check_internet_connection with a successful connection."""
    mock_get = mocker.patch("requests.get", return_value=mocker.Mock(status_code=200))
    result = processor._check_internet_connection(timeout=5)
    assert result is True
    mock_get.assert_called_once()


def test_check_internet_connection_failure(processor, mocker):
    """Test _check_internet_connection with a failed connection."""
    mock_get = mocker.patch("requests.get", side_effect=requests.RequestException("Connection error"))
    result = processor._check_internet_connection(timeout=5)
    assert result is False
    mock_get.assert_called_once_with("https://www.google.com", timeout=5)


def test_test_llm_connection_timeout_with_retry(processor, mocker):
    """Test _test_llm_connection with timeout and retries, eventually succeeding."""
    timeout_error = Timeout("Timeout", model="gpt-4o-mini", llm_provider="openai")
    mock_response = mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content="Success"))])

    mocker.patch("d2d.processor.completion", side_effect=[timeout_error, timeout_error, mock_response])
    mocker.patch("builtins.open", mock_open(read_data='{"openai": "gpt-4o-mini", "anthropic": "claude-3-5-sonnet"}'))
    mocker.patch("time.sleep")

    result = processor._test_llm_connection()
    assert result is True


def test_test_llm_connection_config_file_missing(processor, mocker):
    """Test _test_llm_connection when llm_defaults.json is missing."""
    mock_response = mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content="Success"))])
    mocker.patch("d2d.processor.completion", return_value=mock_response)
    mocker.patch("builtins.open", side_effect=FileNotFoundError)

    result = processor._test_llm_connection()
    assert result is True


def test_test_llm_connection_invalid_json(processor, mocker):
    """Test _test_llm_connection with invalid JSON in llm_defaults.json."""
    mock_response = mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content="Success"))])
    mocker.patch("d2d.processor.completion", return_value=mock_response)
    mocker.patch("builtins.open", mock_open(read_data="invalid json"))

    result = processor._test_llm_connection()
    assert result is True


if __name__ == '__main__':
    print(D2DProcessor.__module__)  # Should output the true import path