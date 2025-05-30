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
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import traceback

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


def find_reference_for_answers(match: dict, extracted_phrase: str = None, embedding_model: SentenceTransformer = None, device: torch.device = "cpu", logger: logging.Logger = None):
    # Looking for reference points for answers in transcript
    response_position = []
    line_reference = set() # init as a set so that partial matches do not get added multiple times
    interviewee_match = []
    match_type = None
    partial_match_threshold = 70 # set a 70% partial fuzzy match minimum with fuzz
    fuzzy_scores = []

    if (not extracted_phrase) or (extracted_phrase == "No relevant response found"):
        # nothing to be added
        logger.info("No matches because extracted phrase is None/[No relevant response found]")
    else:
        # check for fuzzy or exact match on all matched lines
        logger.info(f"{'*'*90}")
        logger.info(f"Matched lines for '{extracted_phrase}'")
        for m in match["matches"]:
            src_string = extracted_phrase.lower()
            dest_string = m['response'].lower()
            fuzzy_match = fuzz.partial_ratio_alignment(src_string, dest_string)
            fuzzy_scores.append({"line": m['interviewee_line_ref'], "score": fuzzy_match.score})
            if fuzzy_match.score == 100:
                # perfect match, add current line
                line_reference.add(m['interviewee_line_ref'])
                # reference is triple containing (line, start_index, end_index)
                response_position.append({'line': m['interviewee_line_ref'],
                                        'start': fuzzy_match.dest_start,
                                        'end': fuzzy_match.dest_end})
                match_type = "EXACT"
                interviewee_match.append(m['response'])
            # we can decide whether to ignore partial matches if we already have an exact
            elif fuzzy_match.score >= partial_match_threshold:
                # partial match, we will extract matching portions
                opcodes = Levenshtein.opcodes(src_string, dest_string)
                match_type = "PARTIAL"
                for tag, i1, i2, j1, j2 in opcodes:
                    if tag == "equal":
                        # we leave out any non-word matches or partial matches shorter than 3 characters
                        if(re.search(r'\w+', extracted_phrase[i1:i2]) and (i2-i1)>=3):
                            logger.info(f"Match: extracted_phrase[{i1}:{i2}] == response[{j1}:{j2}]")
                            logger.info(f"  extracted_phrase: '{extracted_phrase[i1:i2]}'")
                            logger.info(f"  response: '{m['response'][j1:j2]}'")
                            # add this reference
                            line_reference.add(m['interviewee_line_ref'])
                            response_position.append({'line': m['interviewee_line_ref'], 'start': j1, 'end': j2})
                        else:
                            logger.info(f"Very short match: extracted_phrase[{i1}:{i2}] == response[{j1}:{j2}]")
                        logger.info("\n")

            logger.info(f"{str(m['interviewee_line_ref']).rjust(3)} = ({fuzzy_match.score:3.1f}): {m['response']}")

        if len(line_reference) > 0:
            logger.info(f"Line References found using Fuzzy Match at {line_reference}")
        else:
            # Could not find fuzzy matches of extracted phrase from context
            # Approach 2: find references using embedding of interviewee responses vs extracted response
            if embedding_model:
                logger.info(f"Finding semantic matches using embedding...")
                top_response_matches = match_top_responses(embedding_model, device, logger, extracted_phrase,
                                                        match["matches"], 
                                                        p_threshold = partial_match_threshold/100.0)
                for top_match in top_response_matches:
                    line_reference.add(top_match['interviewee_line_ref'])
                    response_position.append({'line': top_match['interviewee_line_ref'], 'start': -1, 'end': -1})
                    interviewee_match.append(top_match['response'])

            # if still not found ...
            if len(line_reference) == 0:
                # assume all matching lines were used
                logger.info(f"Line Reference: No matches found using either method")
                #pick the highest fuzzy score
                max_fuzzy = max(fuzzy_scores, key=lambda s: s['score'])
                line_reference.add(max_fuzzy['line'])
                response_position.append({'line': max_fuzzy['line'], 'start': -1, 'end': -1})
                match_type = "FUZZY"
            else:
                logger.info(f"Line Reference: Semantic Match at {line_reference}")
                match_type = "SEMANTIC"

        logger.info(f"{'*'*90}")

    return (list(line_reference), response_position, interviewee_match, match_type)

async def generate_output_from_summarized_matches_async(transcript_files: list, matches_list: list, guide_questions: list,
                                                        llm_model: str, output_path: str,
                                                        max_concurrent_calls: int = 10,
                                                        logger: logging.Logger = None,
                                                        embedding_model: SentenceTransformer = None,
                                                        device: torch.device = "cpu",
                                                        custom_extract_prompt: str = None,
                                                        custom_summarize_prompt: str = None
                                                        ) -> None:
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
                return_val = await extract_and_summarize_response_llm_async(file_name, context, guide_question, llm_model, logger,
                                                                            custom_extract_prompt=custom_extract_prompt,
                                                                            custom_summarize_prompt=custom_summarize_prompt
                                                                            )
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
                # for l in log:
                #     logger.info(l)
                logger.info(f"Response from generator {return_val}")
                # ===============================================================================================================

                # Find reference points for answers in the transcript
                (line_reference, response_position, interviewee_match, match_type) = find_reference_for_answers(match, extracted_phrase, embedding_model, device, logger)


                if extracted_phrase:
                    logger.info(f"Extracted Phrase was '{extracted_phrase}': Reference (Line={list(zip(line_reference, response_position))})")
                    logger.info(f"Referenced Interviewee Responses: {interviewee_match}")
                else:
                    logger.info(f"Extracted Phrase was '{extracted_phrase}': [No Reference]")
                logger.info(f"Summarized Response for '{guide_question}': {response}")

                output_divider(logger, True)
                relevant_lines = [(m['interviewer_line_ref'], m['interviewee_line_ref']) for m in match["matches"]]
                relevant_lines = sorted(relevant_lines, key=lambda x: x[0])
                return {
                    # SI; I think we might need both interviewer and interviewee segments
                    'relevant_lines': relevant_lines,
                    'extracted_phrase': extracted_phrase,
                    'response': response.strip('\"\''),   
                    'match_type': match_type,                 
                    'extracted_line_references': line_reference if len(line_reference)>0 else None,
                    'extracted_character_index': response_position if len(response_position)>0 else None
                    }, None
            except Exception as e:
                traceback.print_exc()
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
    base_name, ext = os.path.splitext(file_name)
    responses_file_name = f"{base_name}_responses_{timestamp}{ext}"
    references_file_name = f"{base_name}_references_{timestamp}.json"
    generator_log_file_name = f"{base_name}_generator_log_{timestamp}.txt"




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
    output_df.to_csv(os.path.join(dir_name, responses_file_name), index=False)
    logger.info(f"Responses saved to {new_output_path}")
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
    references_file = os.path.join(dir_name, references_file_name)
    with open(references_file, 'w', encoding ='utf8') as file:
        json.dump(reference_data, file, indent=4)
        logger.info(f"JSON output saved to {references_file}")
    # ===============================================================================================================

    return None



