import pandas as pd
import os, sys
import logging
import asyncio
from datetime import datetime
import re
import json
from src.utils.api_utils import extract_and_summarize_response_llm_async

def output_divider(logger: logging.Logger, line_brk: bool = False):
    logger.info(
        f"================================================================================================{'\n' if line_brk else ''}")

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

async def generate_output_from_summarized_matches_async(transcript_files: list, matches_list: list, guide_questions: list,
                                                       gpt_model: str, output_path: str, conciseness: int = 0,
                                                       logger: logging.Logger = None) -> None:
    """
    The Generator of the pipeline
    Generate a structured CSV with one row per interview and columns for guide questions, using summarized question matches.

    Args:
        transcript_files (list[str]): List of transcript file paths.
        matches_list (list[list[dict]]): List of matches for each transcript based on summarized questions.
        guide_questions (list[str]): List of guide questions.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).
        logger (logging.Logger, optional): Logger instance for logging execution information.
    """
    logger.info("Generator processing...")

    output_data = []
    reference_data = []
    semaphore = asyncio.Semaphore(10)  # Limit to 5 concurrent API calls

    async def summarize_with_limit(file_name, context, guide_question, match):
        async with semaphore:
            try:
                return_val = await extract_and_summarize_response_llm_async(context, guide_question, gpt_model, conciseness, logger)
                if isinstance(return_val, str):
                    response = return_val
                    raw_response = None
                else:
                    response = return_val[0]
                    # remove any punctuations or quotes at the start or end of the literal response
                    raw_response = re.sub(r'^\W+|\W+$', '', return_val[1])

                output_divider(logger)
                logger.info(f"Processing file: {file_name}")
                logger.info(f"Processing guide question (top-k matches): {guide_question}")
                logger.info("Relevant Interviewee Responses:")

                # Todo: Can be simply replace this line, but just leave it as it's right now
                # logger.info(context)

                response_position = -1
                line_reference = -1
                for m in match["matches"]:
                    logger.info(f"Interviewer: {m['question']}\nInterviewee: {m['response']}")
                    # check if the raw response reported was extracted from this line
                    if raw_response and response_position < 0:
                        response_position = m['response'].lower().find(raw_response.lower())
                        # if found, we record the line and position
                        if response_position >= 0:
                            line_reference = m['interviewee_line_ref']
                            response_position +=1 # make character index 1-based
                            print(f"Response found at {response_position} in on line {line_reference}")

                logger.info(f"Summarized Response for '{guide_question}': {response}")
                if raw_response:
                    logger.info(f"Original Response was '{raw_response}': Reference (Line={line_reference}, Char={response_position})")
                else:
                    logger.info(f"Original Response was '{raw_response}': [No Reference]")

                output_divider(logger, True)
                return {
                    'response': response.strip('\"\''),
                    'raw_response': raw_response,
                    'line_reference': line_reference if line_reference>=0 else None,
                    'char_index': response_position if response_position>=0 else None
                    }, None
            except Exception as e:
                logger.error(f"Error summarizing response for guide question '{guide_question}': {str(e)}")
                return {
                    'response':"[Error summarizing response]"
                    }, str(e)

    for file_path, matches in zip(transcript_files, matches_list):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        interview_refs = []
        row = {"Interview File": file_name}
        output_divider(logger)
        logger.info(f"Processing file: {file_name}")
        output_divider(logger, True)

        # Initialize row with empty responses for all guide questions
        for question in guide_questions:
            row[question] = ""

        # Prepare tasks for parallel summarization
        tasks = []
        task_metadata = []
        for match in matches:
            guide_question = match["guide_question"]
            context = "\n".join([f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in match["matches"]])
            tasks.append(summarize_with_limit(file_name, context, guide_question, match))
            task_metadata.append(guide_question)

        # Execute summarization tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for guide_question, (response, error) in zip(task_metadata, results):
            if error:
                row[guide_question] = response['response']  # Error message from summarize_with_limit
            else:
                row[guide_question] = response['response']
            response_details = response
            response_details["guide_question"] = guide_question
            interview_refs.append(response_details)

        logger.info(f"File {file_name} processing complete.")
        output_divider(logger, True)
        output_data.append(row)
        reference_data.append({"interview": file_name, "responses": interview_refs})

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

    logger.info(f"Output saved to {new_output_path}")
    logger.info(output_df[["Interview File"] + guide_questions[:2]])

    # save detailed response JSON
    json_file_name = re.sub(r'\.csv$', '.json', new_output_path)
    logger.info(f"JSON output file name is {json_file_name}")
    with open(json_file_name, 'w', encoding ='utf8') as file:
        json.dump(reference_data, file, indent=4)
        logger.info(f"JSON output saved to {json_file_name}")

    return None

# def generate_claude_output(interview_files: list[str], matches_list: list[list[dict]], guide_questions: list[str],
#                            llm_model: str, api_key: str, output_path: str):
#     """
#     Generate a structured CSV with one row per interview and columns for guide questions.
#
#     Args:
#         interview_files (list[str]): List of interview file paths.
#         matches_list (list[list[dict]]): List of matches for each interview.
#         guide_questions (list[str]): List of guide questions.
#         llm_model (str): The Claude LLM model name to use.
#         api_key (str): The Anthropic API key.
#         output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
#     """
#     output_data = []
#     for file_path, matches in zip(interview_files, matches_list):
#         file_name = os.path.splitext(os.path.basename(file_path))[0]
#         row = {"Interview File": file_name}
#         for question in guide_questions:
#             row[question] = ""
#         for match in matches:
#             guide_question = match["guide_question"]
#             # handle case of no match
#             if (len(match["matches"]) > 0):
#                 # retain order of speaking in dialogue
#                 ordered_dialogue = sorted(match["matches"], key=lambda item: item['speaking_round'])
#                 context = "\n".join(
#                     [f"Interviewer: {m['question']}\nInterviewee: {m['response']}" for m in ordered_dialogue])
#                 llm_response = extract_and_summarize_response_claude(context, guide_question, llm_model, api_key).strip(
#                     '\"\'')
#                 row[guide_question] = llm_response
#             else:
#                 row[guide_question] = "###Not found"
#         output_data.append(row)
#
#     output_df = pd.DataFrame(output_data)
#
#     # Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
#     dir_name, file_name = os.path.split(output_path)
#     if file_name:
#         base_name, ext = os.path.splitext(file_name)
#         new_file_name = f"{base_name}_{timestamp}{ext}"
#     else:
#         new_file_name = f"matched_interviews_{timestamp}.csv"
#     if dir_name:
#         os.makedirs(dir_name, exist_ok=True)
#         new_output_path = os.path.join(dir_name, new_file_name)
#     else:
#         new_output_path = new_file_name
#
#     output_df.to_csv(new_output_path, index=False)
#     print(f"Output saved to {new_output_path}")


