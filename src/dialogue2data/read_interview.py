# parse_content.py
# author: Sienko Ikhabi
# date: 2025-05-15

import os
import pandas as pd
import re
import json
from glob import glob

def read_interview(transcript_dir: str, guidelines_path: str, output_path: str):
    try:
        # validate the transcript_dir param
        if not os.path.isdir(transcript_dir):
            raise FileNotFoundError(f"Transcripts folder not found: {transcript_dir}")
        # confirm that the folder is accessible
        os.listdir(transcript_dir)  

        # Validate that the file exists
        if not os.path.isfile(guidelines_path):
            raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
        with open(guidelines_path, 'r'):
            # ensure it is accessible
            pass
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise  # propagate error

    # read the guidelines
    try:
        tmp_data = pd.read_csv(guidelines_path)
        if not "guide_text" in tmp_data.columns:
            raise ValueError(f"Guidelines file must contains 'guide_text' column")
        guidelines = tmp_data["guide_text"].tolist()
    except Exception as e:
        raise ValueError(f"Unable to read the Guidelines file '{guidelines_path}' due to error {e}")
    
    # list of trascript files
    transcript_file_list = glob(os.path.join(transcript_dir, "*.txt"))

    if not transcript_file_list:
        raise FileNotFoundError("No transcript files found in the specified directory")
    
    transcripts_list = []
    max_group_length = 0
    # load each file    
    for index, transcript_file in enumerate(transcript_file_list):
        with open(transcript_file, "r", encoding="utf-8") as f: 
            transcript = f.read()
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
                        "interviewee": [turns[i+1][1]],
                        "speaking_round": i//2
                    })
            max_group_length = max(max_group_length, len(groups))
            transcripts_list.append({
                "interview_index": index,
                "interview_id": os.path.splitext(os.path.basename(transcript_file))[0], 
                "groups": groups})

    # write output to specified file
    with open(output_path, "w") as outfile:    
        outfile.write(json.dumps({"guidelines": guidelines,
                                  "max_group_length": max_group_length, 
                                  "transcripts_list": transcripts_list}, 
                                 indent=4)
        )