import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d import D2DProcessor
from pathlib import Path
import shutil
import pandas as pd
import json
import uuid

### fixture functions

@pytest.fixture
def test_case_files(tmp_path):
    def _copy_matching_case(test_case: str):
        """
        Fixture to copy all files and folders from the 'fixtures/' directory
        that have `test_case` as a substring in their filename or folder name.
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
    Copies matching test case files into a temporary directory and creates an additional empty text file.

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
        extra_file = path / f"interview_{test_case}" /extra_filename
        extra_file.write_text(file_content)

        return path
    return _copy_and_add_file

### test entire pipeline as a blackbox

def test_csv_valid_interview(subtests, test_case_files):
    """
    Tests that loading a simple valid guidelines CSV file works as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """

    test_case = "000"
    temp_folder = test_case_files(test_case)
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()

    # create output
    output_folder = temp_folder / "results"
    output_folder.mkdir()
    
    # Step 1: Initialize the processor to use all defaults except matching method
    processor = D2DProcessor(
        sampling_method=D2DProcessor.SamplingMethod.TOP_P
    )

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=str(temp_folder),
        interview_name=f"interview_{test_case}",
        output_dir=output_folder,
        disable_logging=False
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
        print("\n\n") # separator

        # test the content in the CSV file produced in output
        if(filename.endswith(".csv")):
            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you doing at the moment?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"]=="fe4b129c-a507"]
            with subtests.test("Output CSV row has expected values"):
                assert test_row["How are you doing at the moment?"].str.contains("overwhelmed", case=False).all()
                assert (test_row["What plans do you have for summer?"] == "[No relevant response found]").all()

        # test the content in the JSON file produced in output
        if(filename.endswith(".json")):
            with open(file_path) as out_file:
                data = json.load(out_file)
                with subtests.test("Output contains one interview"):
                    assert len(data)==1 # this file only has 1 interview
                # there should be 2 responses in the file
                with subtests.test("Output for interview has 2 responses"):
                    assert len(data[0]["responses"])==2
                # positive response
                pos_resp = (
                    next((obj for obj in data[0]["responses"] 
                          if obj["guide_question"] == "How are you doing at the moment?"), None)
                )
                with subtests.test("Positive response column exists"):
                    assert pos_resp
                with subtests.test("Positive response references correct line"):
                    assert pos_resp["relevant_lines"]==[7]
                with subtests.test("Positive response extracts correct references"):
                    assert pos_resp["extracted_line_references"]==[7]

                # negatve response
                neg_resp = (
                    next((obj for obj in data[0]["responses"] 
                          if obj["guide_question"] == "What plans do you have for summer?"), None)
                )
                with subtests.test("Negative response column exists"):
                    assert neg_resp
                with subtests.test("Negative response has 0 references"):
                    assert neg_resp["relevant_lines"]==[]
                with subtests.test("Negative response does not extract references"):
                    assert neg_resp["extracted_line_references"] is None
                

def test_csv_valid_interview_plus_empty(subtests, test_case_files_with_extra_file):
    """
    Tests that loading a simple valid guidelines CSV file and an empty transcript work as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """

    test_case = "000"
    extra_file_name = f"{str(uuid.uuid4())}.txt"
    temp_folder = test_case_files_with_extra_file(test_case, 
                                                  extra_filename=extra_file_name,
                                                  file_content="This file is empty.")
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
        data_dir=str(temp_folder),
        interview_name=f"interview_{test_case}",
        output_dir=output_folder,
        disable_logging=False
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
        print("\n\n") # separator

        # test the content in the CSV file produced in output
        if(filename.endswith(".csv")):
            df = pd.read_csv(file_path)
            # check that the columns come from the guidelines
            with subtests.test("Output CSV has required columns"):
                assert "Interview File" in df.columns
                assert "How are you doing at the moment?" in df.columns
                assert "What plans do you have for summer?" in df.columns
            # check that the row for the sample interview is as expected
            test_row = df[df["Interview File"]==extra_file_name]
            with subtests.test("Output CSV row has expected values"):
                assert (test_row["How are you doing at the moment?"] == "[No relevant response found]").all()
                assert (test_row["What plans do you have for summer?"] == "[No relevant response found]").all()