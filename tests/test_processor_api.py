import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d import D2DProcessor
from litellm.exceptions import AuthenticationError


def test_openai_apikey(test_case_files, setup_bad_apikey, subtests):
    """
    Tests that processing the simple valid interview set (guidelines + transcripts) with default settings will
    fail if we apply the setup_bad_apikey fixture to override the OPENAI_API_KEY with an incorrect value.

    Args:
        test_case_files (fixture function): A fixture function that create a set of test files from the fixtures folder
        setup_bad_apikey (function): Fixture function that overrides the  OPENAI_API_KEY in .env
        subtests (Generator): Generator to facilitate multiple tests within this function
    """
    
    # confirm value of API Key
    print(f"Using OpenAI API Key = '{os.getenv('OPENAI_API_KEY')}'")

    # set up default processor
    test_case = "000"
    temp_folder = test_case_files(test_case)
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()

    # create output
    output_folder = temp_folder / "results"
    output_folder.mkdir()
    
    # Step 1: Initialize the processor to use all defaults
    processor = D2DProcessor()

    # Test that the main process will raise an litellm.exceptions.AuthenticationError
    with pytest.raises(AuthenticationError):
        # Step 3: Start transcripts processing
        processor.process_transcripts(
            data_dir=str(temp_folder),
            interview_name=f"interview_{test_case}",
            output_dir=output_folder,
            disable_logging=False
        )
    
    # dump content of all files in the output folder to verify output, if needed "-s flag"
    for filename in os.listdir(str(output_folder)):
        file_path = os.path.join(str(output_folder), filename)
        print(f"--- Contents of {filename} ---")
        with open(file_path, 'r') as file:
            print(file.read())
        print("\n\n") # separator
    
    with subtests.test(f"AuthenticationError should not produce CSV output file"):
        # check that no CSV files are produced
        glob_files = list(output_folder.glob("*.csv"))
        assert not glob_files, f"Unexpected .csv files found: {[f.name for f in glob_files]}"
    with subtests.test(f"AuthenticationError should not produce JSON output file."):
        # check that no JSON files are produced
        glob_files = list(output_folder.glob("*.json"))
        assert not glob_files, f"Unexpected .json files found: {[f.name for f in glob_files]}"

