import pandas as pd
import os, sys
import logging
import asyncio
from datetime import datetime
import re
import json
from sentence_transformers import SentenceTransformer, util
import torch
from src.utils.embedding_utils import match_top_responses
from src.utils.api_utils import extract_and_summarize_response_llm_async


def get_divider(line_brk: bool = False):
    return f"================================================================================================{'\n' if line_brk else ''}"

def output_divider(logger: logging.Logger, line_brk: bool = False):
    logger.info(get_divider(line_brk))

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


def find_reference_for_answers(match: dict, response: str, extracted_phrase: str = None, embedding_model: SentenceTransformer = None, device: torch.device = "cpu", logger: logging.Logger = None):
    # Looking for reference points for answers in transcript
    response_position = []
    line_reference = []
    interviewee_match = []
    match_type = None

    # Approach 1: check if the extracted phrase (if there is) from LLM can be found exactly from context
    for m in match["matches"]:
        if extracted_phrase:
            current_line_match = m['response'].lower().find(extracted_phrase.lower())
            # if found, we record the line and position
            if current_line_match >= 0:
                # make both line and character indexes 1-based
                line_reference.append(m['interviewee_line_ref'] + 1)
                response_position.append(current_line_match + 1)
                match_type = "EXACT"
                interviewee_match.append(m['response'])

    if len(line_reference) > 0:
        logger.info(f"Line Reference: Exact Match at {line_reference}")

    # LLM could not find relevant answers
    elif response == "[No relevant response found]":
        response_position = []
        line_reference = []

    else:
        # Could not find exact matches of extracted phrase from context
        # Approach 2: find references using embedding of interviewee responses vs extracted response
        if embedding_model:
            logger.info(f"Finding semantic matches using embedding...")
            top_response_matches = match_top_responses(embedding_model, device, logger, extracted_phrase,
                                                       match["matches"])
            for top_match in top_response_matches:
                line_reference.append(top_match['interviewee_line_ref'] + 1)
                response_position.append(0)
                interviewee_match.append(top_match['response'])

        # if still not found ...
        if len(line_reference) == 0:
            # assume all matching lines were used
            logger.info(f"Line Reference: No matches found, assuming all references used.")
            line_reference = [m['interviewee_line_ref'] for m in match["matches"]]
            response_position = [0] * len(line_reference)  # no character index, whole line assumed
            interviewee_match = [m['response'] for m in match["matches"]]
            match_type = "ALL_RELEVANT"
        else:
            logger.info(f"Line Reference: Semantic Match at {line_reference}")
            match_type = "SEMANTIC"
    return (line_reference, response_position, interviewee_match, match_type)

async def generate_output_from_summarized_matches_async(transcript_files: list, matches_list: list, guide_questions: list,
                                                        llm_model: str, output_path: str,
                                                        sampling_method: str = "top_k", top_k: int = 3,
                                                        max_concurrent_calls: int = 10,
                                                        logger: logging.Logger = None,
                                                        embedding_model: SentenceTransformer = None,
                                                        device: torch.device = "cpu") -> None:
    """
    The Generator of the pipeline
    Generate a structured CSV with one row per interview and columns for guide questions, using summarized question matches.

    Args:
        transcript_files (list[str]): List of transcript file paths.
        matches_list (list[list[dict]]): List of matches for each transcript based on summarized questions.
        guide_questions (list[str]): List of guide questions.
        llm_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): The base path for the output CSV file. The timestamp will be appended to the filename.
        logger (logging.Logger, optional): Logger instance for logging execution information.
    """
    logger.info("Generator processing...")

    output_data = []
    reference_data = []
    generator_log = [] # Will be output for evaluation module

    semaphore = asyncio.Semaphore(max_concurrent_calls)  # Limit to 5 concurrent API calls

    # ===============================================================================================================
    async def extract_and_summarize_response(file_name, context, guide_question, match):
        async with semaphore:
            try:
                return_val = await extract_and_summarize_response_llm_async(context, guide_question, llm_model, logger)
                if isinstance(return_val, str):
                    response = return_val
                    extracted_phrase = None
                else:
                    response = return_val[0]
                    # remove any punctuations or quotes at the start or end of the literal response if valid response
                    extracted_phrase = (
                        re.sub(r'^\W+|\W+$', '', return_val[1]) 
                        if return_val[1] != "[No relevant response found]"
                        else return_val[1]
                    )


                # ===============================================================================================================
                # Create the generator summary log for evaluation module
                log = []
                log.append("===Start===")
                log.append(f"Processing file: {file_name}")
                log.append(f"Processing guide question: {guide_question}")
                log.append("Relevant Interviewee Responses:")

                for m in match["matches"]:
                    log.append(f"Interviewer: {m['question']}\nInterviewee: {m['response']}")

                log.append("===End===")
                generator_log.extend(log)
                for l in log:
                    logger.info(l)
                # ===============================================================================================================


                (line_reference, response_position, interviewee_match, match_type) = find_reference_for_answers(match, response, extracted_phrase, embedding_model, device, logger)

                logger.info(f"Summarized Response for '{guide_question}': {response}")
                if extracted_phrase:
                    logger.info(f"Extracted Phrase was '{extracted_phrase}': Reference (Line={list(zip(line_reference, response_position))})")
                    logger.info(f"Referenced Interviewee Responses: {interviewee_match}")
                else:
                    logger.info(f"Extracted Phrase was '{extracted_phrase}': [No Reference]")

                output_divider(logger, True)
                return {
                    'relevant_lines': [m['interviewee_line_ref']+1 for m in match["matches"]],
                    'extracted_phrase': extracted_phrase,
                    'response': response.strip('\"\''),   
                    'match_type': match_type,                 
                    'extracted_line_references': line_reference if len(line_reference)>0 else None,
                    'extracted_character_index': response_position if len(response_position)>0 else None
                    }, None
            except Exception as e:
                #traceback.print_exc()
                logger.error(f"Error summarizing response for guide question '{guide_question}': {str(e)}")
                return {
                    'response':"[Error summarizing response]"
                    }, str(e)
    # ===============================================================================================================

    # ===============================================================================================================
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
            tasks.append(extract_and_summarize_response(file_name, context, guide_question, match))
            task_metadata.append(guide_question)

        # Execute summarization tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for guide_question, (response, error) in zip(task_metadata, results):
            if error:
                row[guide_question] = response['response']  # Error message from summarize_with_limit
            else:
                row[guide_question] = response['response']
            # record the response details
            response_details = {"guide_question": guide_question}
            response_details.update(response)

            interview_refs.append(response_details)

        logger.info(f"File {file_name} processing complete.")
        output_divider(logger, True)
        output_data.append(row)
        reference_data.append({"interview": file_name, "responses": interview_refs})

    output_df = pd.DataFrame(output_data)
    # ===============================================================================================================

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # ===============================================================================================================
    dir_name, file_name = os.path.split(output_path)
    if file_name:
        base_name, ext = os.path.splitext(file_name)
        new_file_name = f"{base_name}_{timestamp}{ext}"
        generator_log_file_name = f"{base_name}_generator_log_{timestamp}.txt"
    else:
        new_file_name = f"matched_interviews_summarized_{timestamp}.csv"

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        new_output_path = os.path.join(dir_name, new_file_name)
    else:
        new_output_path = new_file_name
    # ===============================================================================================================

    # ===============================================================================================================
    # Write matched responses to file
    output_df.to_csv(new_output_path, index=False)
    logger.info(f"Output saved to {new_output_path}")
    logger.info(output_df[["Interview File"] + guide_questions[:2]])
    # ===============================================================================================================

    # ===============================================================================================================
    # Write generator log to file
    generator_log_file = open(os.path.join(dir_name, generator_log_file_name), 'w')
    for l in generator_log:
        generator_log_file.write(f"{l}\n")
    generator_log_file.close()
    logger.info(f"Generator log saved to {generator_log_file}")
    # ===============================================================================================================

    # ===============================================================================================================
    # save detailed response JSON
    json_file_name = re.sub(r'\.csv$', '.json', new_output_path)
    with open(json_file_name, 'w', encoding ='utf8') as file:
        json.dump(reference_data, file, indent=4)
        logger.info(f"JSON output saved to {json_file_name}")
    # ===============================================================================================================

    return None



