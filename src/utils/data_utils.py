import pandas as pd
import re

def load_guidelines(guidelines_path: str) -> list[str]:
    """
    Load discussion guide questions from a CSV file.
    
    Args:
        guidelines_path (str): Path to the CSV file containing guidelines.
    
    Returns:
        list[str]: List of guide questions.
    """
    guidelines = pd.read_csv(guidelines_path)
    return guidelines["guide_text"].tolist()

def load_transcript(transcript_path: str) -> str:
    """
    Load a transcript file into a string.
    
    Args:
        transcript_path (str): Path to the transcript file.
    
    Returns:
        str: Content of the transcript file.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read()

def segment_transcript(transcript: str) -> list[dict]:
    """
    Segment the transcript into question-response pairs.
    
    Args:
        transcript (str): The transcript text.
    
    Returns:
        list[dict]: List of dictionaries containing interviewer and interviewee turns and their line reference number (0-indexed).
    """
    list_interviewer = []
    list_interviewee = []
    lines = transcript.split(sep='\n', )
    for line_number, line in enumerate(lines):
        # strip and remove any whitespace at start or finish
        line = line.strip()
        if line.startswith("Interviewer:"):
            # for backward compatibility, I append both the text and line_number, which will be separated in the dict
            # All line-refs are human-readable (start at index 1)
            list_interviewer.append( 
                (line[len('Interviewer:'):].strip(), line_number+1)
                )  
        elif line.startswith("Interviewee:"):
            list_interviewee.append(
                (line[len('Interviewee:'):].strip(), line_number+1)
                ) 
        # for now we skip blank lines or those not marked with either speaker, no else
    
    groups = []
    speaking_round = 0
    for qa_pair in zip(list_interviewer, list_interviewee):
        groups.append({
            "interviewer": [qa_pair[0][0]],
            "interviewee": [qa_pair[1][0]],
            "interviewer_line_ref": qa_pair[0][1],
            "interviewee_line_ref": qa_pair[1][1],
            "speaking_round": speaking_round
        })
        speaking_round +=1
    return groups
    