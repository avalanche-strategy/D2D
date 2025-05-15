import re
import pandas as pd
from pathlib import Path

def extract_retrieved_contexts(log_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Extract retrieved interview contexts from a log file.

    Parameters:
    - log_path (str): Path to the .log file.
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
    current_id = None
    current_question = None
    collecting = False
    context_lines = []

    for line in lines:
        line = line.strip()

        m_file = re.match(r".*Processing file: ([a-f0-9\-]+)", line)
        if m_file:
            current_id = m_file.group(1)
            continue

        m_question = re.match(r".*Processing guide question \(top-k mataches\): (.+)", line)
        if m_question:
            if current_id and current_question and context_lines:
                clean_context = [l for l in context_lines if l.startswith("Interviewer:") or l.startswith("Interviewee:")]
                if clean_context:
                    records.append({
                        "respondent_id": current_id,
                        "guide_question": current_question,
                        "retrieved_context": "\n".join(clean_context)
                    })
            current_question = m_question.group(1)
            context_lines = []
            collecting = False
            continue

        if "Relevant Interviewee Responses:" in line:
            collecting = True
            continue

        if collecting:
            if "Processing guide question" in line or "Processing file:" in line:
                collecting = False
                continue
            if ("Summarized Response" in line or
                "File id processing complete." in line or
                "Output saved to" in line):
                continue
            line = re.sub(r"^\d{4}-\d{2}-\d{2}.*? - INFO - ", "", line)
            if line and not re.match(r"=+", line):
                context_lines.append(line)

    if current_id and current_question and context_lines:
        clean_context = [l for l in context_lines if l.startswith("Interviewer:") or l.startswith("Interviewee:")]
        if clean_context:
            records.append({
                "respondent_id": current_id,
                "guide_question": current_question,
                "retrieved_context": "\n".join(clean_context)
            })

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)

    return df
