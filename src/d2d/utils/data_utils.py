import pandas as pd
import re
from itertools import zip_longest

def load_guidelines(guidelines_path: str) -> list[str]:
    """
    Load discussion guide questions from a CSV file.

    Args:
        guidelines_path (str): Path to the CSV file containing guidelines.

    Returns:
        list[str]: List of guide questions from the 'guide_text' column.

    Raises:
        ValueError: If the CSV file lacks a 'guide_text' column or contains no questions.
    """
    guidelines = pd.read_csv(guidelines_path)
    
    if not ("guide_text" in guidelines.columns):
        raise ValueError("Guidelines CSV file must contain the 'guide_text' column.")
    guidelines_list = guidelines[guidelines["guide_text"].notna()]["guide_text"].tolist()
    if len(guidelines_list)==0:
        raise ValueError("Guidelines CSV file must contain at least one question.")
    return guidelines_list

def load_transcript(transcript_path: str) -> str:
    """
    Load a transcript file into a string.

    Args:
        transcript_path (str): Path to the transcript file.

    Returns:
        str: Content of the transcript file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If an error occurs while reading the file.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read()

def segment_transcript(transcript: str) -> list[dict]:
    """
    Segment the transcript into question-response pairs.

    Args:
        transcript (str): The transcript text.

    Returns:
        list[dict]: List of dictionaries, each containing:
            - interviewer (list[str]): The interviewer's turn.
            - interviewee (list[str]): The interviewee's turn.
            - interviewer_line_ref (int): Line number (1-indexed) of the interviewer's turn, or -1 if empty.
            - interviewee_line_ref (int): Line number (1-indexed) of the interviewee's turn, or -1 if empty.
            - speaking_round (int): The index of the conversation round (0-indexed).
    """
    list_interviewer = []
    list_interviewee = []
    lines = transcript.split(sep='\n', )
    previous_type = None
    for line_number, line in enumerate(lines):
        # strip and remove any whitespace at start or finish
        line = line.strip()
        if line.startswith("Interviewer:"):
            # if the previous line was also Interviewer, we need to add a blank line for interviewee
            if previous_type == "Interviewer:":
                list_interviewee.append(("", -1))
            # for backward compatibility, I append both the text and line_number, which will be separated in the dict
            # All line-refs are human-readable (start at index 1)
            list_interviewer.append((line[len('Interviewer:'):].strip(), line_number+1))
            previous_type = "Interviewer:"
        elif line.startswith("Interviewee:"):
            # append interviewer blank line if consecutive interviewee
            if previous_type == "Interviewee:":
                list_interviewer.append(("", -1))
            list_interviewee.append(
                (line[len('Interviewee:'):].strip(), line_number+1)
                ) 
            previous_type = "Interviewee:"
        # for now we skip blank lines or those not marked with either speaker, no else statement
    
    groups = []
    speaking_round = 0
    for qa_pair in zip_longest(list_interviewer, list_interviewee, fillvalue=("", -1)):
        groups.append({
            "interviewer": [qa_pair[0][0]],
            "interviewee": [qa_pair[1][0]],
            "interviewer_line_ref": qa_pair[0][1],
            "interviewee_line_ref": qa_pair[1][1],
            "speaking_round": speaking_round
        })
        speaking_round +=1
    return groups
    