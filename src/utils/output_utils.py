import pandas as pd
import os
from datetime import datetime
from src.utils.api_utils import extract_and_summarize_response_chatgpt

def generate_output(interview_files: list[str], matches_list: list[list[dict]], guide_questions: list[str], gpt_model: str, api_key: str, output_path: str):
    """
    Generate a structured CSV with one row per interview and columns for guide questions.
    
    Args:
        interview_files (list[str]): List of interview file paths.
        matches_list (list[list[dict]]): List of matches for each interview.
        guide_questions (list[str]): List of guide questions.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
    """
    output_data = []
    for file_path, matches in zip(interview_files, matches_list):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        row = {"Interview File": file_name}
        for question in guide_questions:
            row[question] = ""
        for match in matches:
            guide_question = match["guide_question"]
            context = "\n".join([f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in match["matches"]])
            chatgpt_response = extract_and_summarize_response_chatgpt(context, guide_question, gpt_model, api_key).strip('\"\'')
            row[guide_question] = chatgpt_response
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dir_name, file_name = os.path.split(output_path)
    if file_name:
        base_name, ext = os.path.splitext(file_name)
        new_file_name = f"{base_name}_{timestamp}{ext}"
    else:
        new_file_name = f"matched_interviews_{timestamp}.csv"
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        new_output_path = os.path.join(dir_name, new_file_name)
    else:
        new_output_path = new_file_name
    
    output_df.to_csv(new_output_path, index=False)
    print(f"Output saved to {new_output_path}")
    print(output_df[["Interview File"] + guide_questions[:2]])


def generate_output_from_summarized_matches(interview_files: list[str], matches_list: list[list[dict]], guide_questions: list[str],
                               gpt_model: str, api_key: str, output_path: str):
    """
    Generate a structured CSV with one row per interview and columns for guide questions, using summarized question matches.

    Args:
        interview_files (list[str]): List of interview file paths.
        matches_list (list[list[dict]]): List of matches for each interview based on summarized questions.
        guide_questions (list[str]): List of guide questions.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
    """
    output_data = []
    for file_path, matches in zip(interview_files, matches_list):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        row = {"Interview File": file_name}
        for question in guide_questions:
            row[question] = ""
        for match in matches:
            guide_question = match["guide_question"]
            context = "\n".join(
                [f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in match["matches"]])
            chatgpt_response = extract_and_summarize_response_chatgpt(context, guide_question, gpt_model, api_key).strip('\"\'')
            row[guide_question] = chatgpt_response
        output_data.append(row)

    output_df = pd.DataFrame(output_data)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dir_name, file_name = os.path.split(output_path)
    if file_name:
        base_name, ext = os.path.splitext(file_name)
        new_file_name = f"{base_name}_{timestamp}{ext}"
    else:
        new_file_name = f"matched_interviews_summarized_{timestamp}.csv"
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        new_output_path = os.path.join(dir_name, new_file_name)
    else:
        new_output_path = new_file_name

    output_df.to_csv(new_output_path, index=False)
    print(f"Output saved to {new_output_path}")
    print(output_df[["Interview File"] + guide_questions[:2]])