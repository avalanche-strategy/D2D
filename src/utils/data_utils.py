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
        list[dict]: List of dictionaries containing interviewer and interviewee turns.
    """
    parts = re.split(r'(Interviewer:|Interviewee:)', transcript)
    turns = []
    current_speaker = None
    current_text = ""
    for part in parts:
        if part in ["Interviewer:", "Interviewee:"]:
            if current_speaker:
                turns.append((current_speaker, current_text.strip()))
            current_speaker = part
            current_text = ""
        else:
            current_text += part
    if current_speaker:
        turns.append((current_speaker, current_text.strip()))
    groups = []
    for i in range(0, len(turns) - 1, 2):
        if turns[i][0] == "Interviewer:" and turns[i+1][0] == "Interviewee:":
            groups.append({
                "interviewer": [turns[i][1]],
                "interviewee": [turns[i+1][1]]
            })
    return groups