# test_read_interview.py
# author: Sienko Ikhabi
# date: 2025-05-15

import pytest
import json
#from dialogue2data import read_interview
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dialogue2data.read_interview import read_interview

simple_guide_csv = """guide_text
"Will this test pass?"
"Can you confirm that this is second?"
"""

simple_transcript = """
Interviewer: Hello and thanks for speaking with me today.

Interviewee: No problem at all!
"""

empty_single_file = {
  "guidelines": [
    "Will this test pass?",
    "Can you confirm that this is second?"
  ],
  "max_group_length": 0,
  "transcripts_list": [
    {
      "interview_index": 0,
      "interview_id": "transcript1",
      "groups": []
    }
  ]
}

def test_nonexistent_file_raises(tmp_path):
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()
    temp_file = os.path.join(temp_folder, 'some_fake_file.txt')
    with pytest.raises(FileNotFoundError):
        read_interview(str(temp_folder), str(temp_file), os.path.join(temp_folder, 'output.json'))

def test_file_instead_of_folder_raises(tmp_path):
    temp_file = tmp_path / "text_file.txt"
    temp_file.write_text("This is a file that will be passed as a folder.")
    
    with pytest.raises(FileNotFoundError):
        read_interview(str(temp_file), str(temp_file), os.path.join(tmp_path, 'output.json'))

def test_file_csv_no_guide_text_raises(tmp_path):
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()
    temp_file = temp_folder / "input.csv"
    temp_file.write_text("This file does not have the 'guide_text' column.")
    
    with pytest.raises(ValueError, match="Guidelines file must contains 'guide_text' column"):
        read_interview(str(temp_folder), str(temp_file), os.path.join(temp_folder, 'output.json'))

def test_empty_transcript_folder_raises(tmp_path):
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()
    temp_file = temp_folder / "input.csv"
    temp_file.write_text(simple_guide_csv, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No transcript files found in the specified directory"):
        read_interview(str(temp_folder), str(temp_file), os.path.join(tmp_path, 'output.json'))

def test_valid_csv_empty_transcript_output(tmp_path):
    
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()
    temp_file = temp_folder / "input.csv"
    output_filename = "output.json"
    temp_file.write_text(simple_guide_csv, encoding="utf-8")
    transcript1 = temp_folder / "transcript1.txt"
    transcript1.write_text("", encoding="utf-8")
    
    read_interview(str(temp_folder), str(temp_file), os.path.join(temp_folder, output_filename))

    # check that the output was produced
    written_file = temp_folder / output_filename
    assert written_file.exists()
    # validate file content!
    file_content = written_file.read_text()
    assert json.loads(file_content) == empty_single_file

def test_valid_csv_simple_transcript_output(tmp_path):
    
    temp_folder = tmp_path / "data"
    temp_folder.mkdir()
    temp_file = temp_folder / "input.csv"
    output_filename = "output.json"
    temp_file.write_text(simple_guide_csv, encoding="utf-8")
    transcript1 = temp_folder / "transcript1.txt"
    transcript1.write_text(simple_transcript, encoding="utf-8")
    transcript2 = temp_folder / "transcript2.txt"
    transcript2.write_text(simple_transcript, encoding="utf-8")
    
    read_interview(str(temp_folder), str(temp_file), os.path.join(temp_folder, output_filename))

    # check that the output was produced
    written_file = temp_folder / output_filename
    assert written_file.exists()
    # validate file content!
    #assert written_file.read_text() == "dummy"


