import re
import pandas as pd
from pathlib import Path

def extract_retrieved_contexts(log_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Extract retrieved interview contexts from a log file in block format (===Start=== to ===End===).
    
    Parameters:
    - log_path (str): Path to the .txt log file.
    - save_path (str, optional): If provided, saves the extracted DataFrame to CSV.

    Returns:
    - pd.DataFrame: A DataFrame with columns: respondent_id, guide_question, retrieved_context.
    """
    log_path = Path(log_path).expanduser()
    if save_path:
        save_path = Path(save_path).expanduser()

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    block = []
    in_block = False

    for line in lines:
        line = line.strip()
        if line == "===Start===":
            in_block = True
            block = []
            continue
        elif line == "===End===":
            in_block = False
            respondent_id = None
            guide_question = None
            context_lines = []
            for bline in block:
                if bline.startswith("Processing file:"):
                    respondent_id = bline.replace("Processing file:", "").strip()
                elif bline.startswith("Processing guide question (top-k matches):"):
                    guide_question = bline.replace("Processing guide question (top-k matches):", "").strip()
                elif bline.startswith("Relevant Interviewee Responses:"):
                    context_lines = []  # start collecting after this line
                elif bline.startswith("Interviewer:") or bline.startswith("Interviewee:"):
                    context_lines.append(bline)
            if respondent_id and guide_question and context_lines:
                records.append({
                    "respondent_id": respondent_id,
                    "guide_question": guide_question,
                    "retrieved_context": "\n".join(context_lines)
                })
            block = []
        elif in_block:
            block.append(line)

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)

    return df