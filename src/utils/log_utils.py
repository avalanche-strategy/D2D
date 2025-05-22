import re
import pandas as pd
from pathlib import Path

def extract_retrieved_contexts(log_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Extract retrieved interview contexts from a log file in block format (===Start=== to ===End===),
    and number the retrieved chunks (chunk 1, chunk 2, ...).
    
    Returns a DataFrame with columns: respondent_id, guide_question, retrieved_context.
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
                    context_lines = []  # reset and start collecting actual dialogue
                elif bline.startswith("Interviewer:") or bline.startswith("Interviewee:"):
                    context_lines.append(bline)

            # Number the chunks by pairing interviewer–interviewee lines
            if respondent_id and guide_question and context_lines:
                chunks = []
                i = 0
                chunk_id = 1
                while i < len(context_lines) - 1:
                    if context_lines[i].startswith("Interviewer:") and context_lines[i+1].startswith("Interviewee:"):
                        chunk = f"chunk {chunk_id}:\n{context_lines[i]}\n{context_lines[i+1]}"
                        chunks.append(chunk)
                        chunk_id += 1
                        i += 2
                    else:
                        # If lines are out of expected order, skip one line
                        i += 1

                retrieved_context = "\n\n".join(chunks)
                records.append({
                    "respondent_id": respondent_id,
                    "guide_question": guide_question,
                    "retrieved_context": retrieved_context
                })
            block = []
        elif in_block:
            block.append(line)

    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)

    return df
