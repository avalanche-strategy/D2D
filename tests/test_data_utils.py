import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d.utils.data_utils import load_guidelines, load_transcript, segment_transcript
import pandas as pd


### guidelines file test cases

def test_csv_valid_guidelines(subtests, test_case_files):
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
    
    guidelines_data = load_guidelines(str(guidelines_file))

    with subtests.test(f"Guidelines case {test_case} has 2 questions"):
        assert len(guidelines_data) == 2
    with subtests.test(f"Guidelines case {test_case} Question 1"):
        assert guidelines_data[0] == "How are you doing at the moment?"
    with subtests.test(f"Guidelines case {test_case} Question 2"):
        assert guidelines_data[1] == "What plans do you have for summer?"

def test_nonexistent_guidelines_file_raises(tmp_path):
    """
    Tests that providing a nonexistent argument file raises a FileNotFoundError error.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    guidelines_file = tmp_path / f"interview_guidelines.csv"
    with pytest.raises(FileNotFoundError):
        _ = load_guidelines(str(guidelines_file))

def test_csv_empty_guidelines_raises(tmp_path):
    """
    Tests that providing an argument of an empty file raises a pandas.errors.EmptyDataError error.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    guidelines_file = tmp_path / f"interview_guidelines.csv"
    guidelines_file.write_text("", encoding="utf-8")
    # the file is empty and should raise an error
    with pytest.raises(pd.errors.EmptyDataError):
        _ = load_guidelines(str(guidelines_file))

def test_csv_headeronly_guidelines_raises(tmp_path):
    """
    Tests that providing an argument of a file with just the CSV headers
    raises a ValueError error with a predefined message.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    guidelines_file = tmp_path / f"interview_guidelines.csv"
    # add only a header
    guidelines_file.write_text("guide_text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Guidelines CSV file must contain at least one question."):
        _ = load_guidelines(str(guidelines_file))

def test_csv_guidelines_guide_text_col_missing_raises(test_case_files):
    """
    Tests that loading a guidelines CSV file that does not have a guide_text
    column will raise a ValueError error with a predefined message.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "003"
    temp_folder = test_case_files(test_case)
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"

    with pytest.raises(ValueError, match="Guidelines CSV file must contain the 'guide_text' column."):
        _ = load_guidelines(str(guidelines_file))

def test_binary_guidelines_file_raises(tmp_path):
    """
    Tests that providing an argument of a file with binary data raises a UnicodeDecodeError 
    error (from pd.read_csv).

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest. A binary file
        with random data will be created here.
    """
    guidelines_file = tmp_path / f"interview_guidelines.csv"    
    # write random content into file in binary mode
    with open(guidelines_file, 'wb') as fbin:
        fbin.write(os.urandom(512))

    with pytest.raises(UnicodeDecodeError):
        _ = load_guidelines(str(guidelines_file))


def test_csv_valid_guidelines_more_cols(subtests, test_case_files):
    """
    Tests that loading a guidelines CSV file that contains `guide_text` and other 
    additional columns works as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "001"
    temp_folder = test_case_files(test_case)
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()
    
    guidelines_data = load_guidelines(str(guidelines_file))

    with subtests.test(f"Guidelines case {test_case} has 2 questions"):
        assert len(guidelines_data) == 2
    with subtests.test(f"Guidelines case {test_case} Question 1"):
        assert guidelines_data[0] == "How are you doing at the moment?"
    with subtests.test(f"Guidelines case {test_case} Question 2"):
        assert guidelines_data[1] == "What plans do you have for summer?"

def test_csv_valid_guidelines_with_nan(subtests, test_case_files):
    """
    Tests that loading a guidelines CSV file that contains `guide_text` and other 
    additional columns, where `guide_text` has some rows with missing values.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "002"
    temp_folder = test_case_files(test_case)
    guidelines_file = temp_folder / f"interview_{test_case}_guidelines.csv"
    assert guidelines_file.exists()
    
    guidelines_data = load_guidelines(str(guidelines_file))
    print(guidelines_data)

    with subtests.test(f"Guidelines case {test_case} has 2 questions"):
        assert len(guidelines_data) == 2
    with subtests.test(f"Guidelines case {test_case} Question 1"):
        assert guidelines_data[0] == "Can you confirm that this remains first?"
    with subtests.test(f"Guidelines case {test_case} Question 2"):
        assert guidelines_data[1] == "Is this topic 5 or topic 2?"

### load_transcript

def test_valid_transcipt_file(subtests, test_case_files):
    """
    Tests that loading a simple valid interview transcript works as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_data = load_transcript(str(transcript_file))
    lines = interview_data.split(sep='\n', )

    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' has 8 lines."):
        assert len(lines) == 8
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' Line 0."):
        assert lines[0] == "Interviewer: Hello and thanks for speaking with me today."
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' Line 2."):
        assert lines[2] == "Interviewee: No problem at all!"
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' Line 5."):
        assert lines[5] == ""


def test_nonexistent_transcript_file_raises(tmp_path):
    """
    Tests that providing a nonexistent file raises a FileNotFoundError error.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    transcript_file = tmp_path / f"transcript1.txt"
    with pytest.raises(FileNotFoundError):
        _ = load_guidelines(str(transcript_file))

def test_empty_transcript_file_isvalid(tmp_path):
    """
    Tests that providing an argument of an empty file reads and empty string.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    transcript_file = tmp_path / f"transcript1.txt"
    transcript_file.write_text("", encoding="utf-8")
    interview_data = load_transcript(str(transcript_file))
    assert interview_data == ""

def test_binary_transcript_file_raises(tmp_path):
    """
    Tests that providing an argument of a file with binary data raises an error.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest. A binary file
        with random data will be created here.
    """
    transcript_file = tmp_path / f"binary_transcript.txt"    
    # write random content into file in binary mode
    with open(transcript_file, 'wb') as fbin:
        fbin.write(os.urandom(512))

    with pytest.raises(UnicodeDecodeError):
        _ = load_transcript(str(transcript_file))

### segment_transcript

def test_segment_valid_transcipt_file(subtests, test_case_files):
    """
    Tests that segmenting a simple valid interview transcript works as expected.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "000"
    test_transcript_file = "fe4b129c-a507.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' has 2 QA pairs."):
        assert len(interview_data) == 2
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' QA Pair 0."):
        assert interview_data[0]["interviewer"] == ["Hello and thanks for speaking with me today."]
        assert interview_data[0]["interviewee"] == ["No problem at all!"]
        assert interview_data[0]["speaking_round"] == 0
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' Line Number for QA Pair 1"):
        assert interview_data[1]["interviewer_line_ref"] == 5
        assert interview_data[1]["interviewee_line_ref"] == 7
        assert interview_data[1]["speaking_round"] == 1

def test_segment_empty_transcript_file(tmp_path):
    """
    Tests that providing an argument of an empty file returns an empty array of segments.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    """
    transcript_file = tmp_path / f"transcript1.txt"
    transcript_file.write_text("", encoding="utf-8")
    interview_data = load_transcript(str(transcript_file))
    assert len(interview_data) == 0

def test_segment_valid_consecutive_interviewer(subtests, test_case_files):
    """
    Tests that segmenting a valid interview transcript that has consecutive "Interviewer"
    lines inserts and empty "Interviewee" line between them.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "001"
    test_transcript_file = "ac015e4f-60f8.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    interview_data = segment_transcript(interview_file_content)
    
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' has 2 QA pairs."):
        assert len(interview_data) == 2
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' QA Pair 0."):
        assert interview_data[0]["interviewer"] == ["Hello and thanks for speaking with me today."]
        assert interview_data[0]["interviewee"] == [""]
        assert interview_data[0]["speaking_round"] == 0
        assert interview_data[0]["interviewee_line_ref"] == -1
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' Line Number for QA Pair 1"):
        assert interview_data[1]["interviewer_line_ref"] == 3
        assert interview_data[1]["interviewee_line_ref"] == 5
        assert interview_data[1]["speaking_round"] == 1
import json
def test_segment_valid_without_interviewee(subtests, test_case_files):
    """
    Tests that segmenting a valid interview transcript that has consecutive "Interviewee"
    lines without ANY "Interviewee" lines creates empty "Interviewee" lines for all segments.

    Args:
        test_case_files (fixture function): A fixture function that create a
        temporary pathlib.Path from fixtures directory
    """
    test_case = "001"
    test_transcript_file = "a7cb-424470af45ac.txt"
    temp_folder = test_case_files(test_case)
    transcript_file = temp_folder / f"interview_{test_case}" / test_transcript_file
    assert transcript_file.exists()
    
    interview_file_content = load_transcript(str(transcript_file))
    print(interview_file_content)
    interview_data = segment_transcript(interview_file_content)
    print(json.dumps(interview_data))
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' has 3 QA pairs."):
        assert len(interview_data) == 3
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' all Interviewee blank."):
        assert all(l["interviewee"]==[""] for l in interview_data)
    with subtests.test(f"Transcript case {test_case} file '{test_transcript_file}' all interviewee_line_ref==-1."):
        assert all(l["interviewee_line_ref"]==-1 for l in interview_data)

# test starting with Interviewee; Different lengths; Unmarked lines (); Tags different from Interviewer/Interviewee

