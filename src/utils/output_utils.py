import pandas as pd
import os, sys
import logging
from datetime import datetime
from src.utils.api_utils import extract_and_summarize_response_chatgpt, extract_and_summarize_response_claude


# Configure logging
def setup_logging(pipeline_name: str, output_path: str, disable_logging: bool = False):
    """
    Set up logging to output to both console and a timestamped log file.

    Args:
        output_path (str): The base path for the output CSV file, used to determine the log file path.
    """
    # Create a logger
    logger = logging.getLogger(pipeline_name)
    if disable_logging:
        # Disable logging by setting the level above CRITICAL
        logger.setLevel(logging.CRITICAL + 1)
        # No handlers needed since no messages will be processed
        return logger

    # Normal logging setup if isn't disabled
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO and above on the console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dir_name, file_name = os.path.split(output_path)
    if file_name:
        base_name, ext = os.path.splitext(file_name)
        log_file_name = f"{base_name}_log_{timestamp}.log"
    else:
        log_file_name = f"{pipeline_name}_log_{timestamp}.log"
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        log_file_path = os.path.join(dir_name, log_file_name)
    else:
        log_file_path = log_file_name

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Write DEBUG and above to the log file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def generate_output(interview_files: list[str], matches_list: list[list[dict]], guide_questions: list[str],
                    gpt_model: str, api_key: str, output_path: str, conciseness=1, logger: logging.Logger = None):
    """
    Generate a structured CSV with one row per interview and columns for guide questions.

    Args:
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).
        interview_files (list[str]): List of interview file paths.
        matches_list (list[list[dict]]): List of matches for each interview.
        guide_questions (list[str]): List of guide questions.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
        logger (logging.Logger, optional): Logger instance for logging execution information.
            Defaults to None.
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
            chatgpt_response = extract_and_summarize_response_chatgpt(context, guide_question, gpt_model, api_key,
                                                                      conciseness, logger).strip('\"\'')
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
    logger.info(f"Output saved to {new_output_path}")
    logger.info(output_df[["Interview File"] + guide_questions[:2]])


def generate_output_from_summarized_matches(interview_files: list[str], matches_list: list[list[dict]],
                                            guide_questions: list[str],
                                            gpt_model: str, api_key: str, output_path: str, conciseness=1,
                                            logger: logging.Logger = None):
    """
    The Generator of the pipeline
    Generate a structured CSV with one row per interview and columns for guide questions, using summarized question matches.

    Args:
        interview_files (list[str]): List of interview file paths.
        matches_list (list[list[dict]]): List of matches for each interview based on summarized questions.
        guide_questions (list[str]): List of guide questions.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
        logger (logging.Logger, optional): Logger instance for logging execution information.
            Defaults to None.
    """
    logger.info("Generator processing...")

    output_data = []
    for file_path, matches in zip(interview_files, matches_list):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        row = {"Interview File": file_name}
        logger.info("================================================================================================")
        logger.info(f"Processing file: {file_name}")
        logger.info(
            "================================================================================================\n")
        for question in guide_questions:
            row[question] = ""
        for match in matches:
            logger.info(
                "================================================================================================")
            guide_question = match["guide_question"]
            logger.info(f"Processing file: {file_name}")
            logger.info(f"Processing guide question (top-k mataches): {guide_question}")
            logger.info("Relevant Interviewee Responses:")
            for m in match["matches"]:
                logger.info(f"Interviewer: {m['question']}\nInterviewee: {m['response']}")
            context = "\n".join(
                [f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in match["matches"]])
            chatgpt_response = extract_and_summarize_response_chatgpt(context, guide_question, gpt_model, api_key,
                                                                      conciseness, logger).strip('\"\'')
            logger.info(f"Summarized Response: {chatgpt_response}")
            logger.info(
                "================================================================================================\n")
            row[guide_question] = chatgpt_response
        logger.info(f"File {file_name} processing complete.")
        logger.info(
            "================================================================================================\n")
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


def generate_claude_output(interview_files: list[str], matches_list: list[list[dict]], guide_questions: list[str],
                           llm_model: str, api_key: str, output_path: str):
    """
    Generate a structured CSV with one row per interview and columns for guide questions.

    Args:
        interview_files (list[str]): List of interview file paths.
        matches_list (list[list[dict]]): List of matches for each interview.
        guide_questions (list[str]): List of guide questions.
        llm_model (str): The Claude LLM model name to use.
        api_key (str): The Anthropic API key.
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
            # handle case of no match
            if (len(match["matches"]) > 0):
                # retain order of speaking in dialogue
                ordered_dialogue = sorted(match["matches"], key=lambda item: item['speaking_round'])
                context = "\n".join(
                    [f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in ordered_dialogue])
                llm_response = extract_and_summarize_response_claude(context, guide_question, llm_model, api_key).strip(
                    '\"\'')
                row[guide_question] = llm_response
            else:
                row[guide_question] = "###Not found"
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

    logger.info(f"Output saved to {new_output_path}")
    logger.info(output_df[["Interview File"] + guide_questions[:2]])

