import logging
from litellm import completion, acompletion
from dotenv import load_dotenv, find_dotenv
from .processor_prompt_utils import build_extract_prompt, build_summarize_prompt

async def extract_and_summarize_response_llm_async(file_name: str, context: str, query: str, llm_model: str,
                                                   logger: logging.Logger = None,
                                                   custom_extract_prompt: str = None,
                                                   custom_summarize_prompt: str = None,
                                                   api_timeout: int = 30
                                                   ) -> str:
    """
    The core function of the Generator in the pipeline.
    Query LLM with the provided context and query asynchronously.

    Args:
        file_name (str): The name of the file being processed.
        context (str): The dialogue context.
        query (str): The query to answer.
        llm_model (str): The LLM model to use.
        logger (logging.Logger, optional): Logger instance for logging execution information.
        custom_extract_prompt (str, optional): Custom prompt template for extracting responses.
        custom_summarize_prompt (str, optional): Custom prompt template for summarizing responses.
        api_timeout (int, optional): Timeout for the API call in seconds (default: 30).

    Returns:
        tuple[str, str] | str: A tuple of (summarized phrase, extracted phrase) on success, or a string
        containing "[No relevant response found]" or an error message if an error occurs.
    """

    # Import here to avoid circular imports
    from .output_utils import output_divider

    _ = load_dotenv(find_dotenv())

    # Define a detailed system prompt for both extraction and summarization
    extraction_system_prompt = """You are a data extraction specialist tasked with processing interview transcripts. 
                           Your role is to extract and summarize information accurately and concisely. When referring to a person, 
                           use their name explicitly and avoid pronouns like "he", "she", "him", "her", or "it/they". Exclude filler 
                           words (e.g., "um", "well", "you know", "like") and irrelevant commentary, retaining only the core content 
                           needed for clarity. """

    summarize_system_prompt = """You are a data summarization specialist tasked with processing extracted phrases from interview transcripts. 
                                 Your role is to summarize information concisely while preserving the primary meaning. When referring to a person, 
                                 use their name explicitly and avoid pronouns like "he", "she", "him", "her", or "it/they". Ensure outputs 
                                 are concise, coherent phrases suitable for a CSV format."""

    # First call: Extract the core phrase from the interviewee's response
    try:
        if len(context.strip()) == 0:
            # no matches using p
            logger.info("Line Referencing: No matching dialog, appending [No relevant response found]")
            # this could be a different string, if we need to debug/differentiate
            return "[No relevant response found]"

        extract_prompt = build_extract_prompt(context, query, custom_extract_prompt)
        response = await acompletion(
            model=llm_model,
            messages=[
                {"role": "system", "content": extraction_system_prompt},
                {"role": "user", "content": extract_prompt}
            ],
            temperature=0,
            timeout=api_timeout
        )
        extracted_phrase = response.choices[0].message.content
        summarize_prompt = build_summarize_prompt(extracted_phrase, query, custom_summarize_prompt)

        # Second call: Summarize the extracted phrase
        try:
            if "[No relevant response found]" in extracted_phrase:
                return ("[No relevant response found]", extracted_phrase)

            response = await acompletion(
                model=llm_model,
                messages=[
                    {"role": "system", "content": summarize_system_prompt},
                    {"role": "user", "content": summarize_prompt}
                ],
                temperature=0,
                timeout=api_timeout
            )
            summarized_phrase = response.choices[0].message.content.strip('\"\'')

            output_divider(logger)
            logger.info(f"Processing file: {file_name}")
            logger.info(f"Extract Prompt: \n{extract_prompt}")
            logger.info(f"Extracted Phrase: [{extracted_phrase}]")
            logger.info(f"Summarize Prompt: \n{summarize_prompt}")
            logger.info(f"Summarize Phrase: [{summarized_phrase}]")
            output_divider(logger)

            return (summarized_phrase, extracted_phrase)

        except Exception as e:
            logger.error(f"Error summarizing response: {str(e)}")
            return f"Error querying ChatGPT: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting response: {str(e)}")
        return f"Error querying ChatGPT: {str(e)}"


async def summarize_question_async(question: str, llm_model: str, logger: logging.Logger = None,
                                   api_timeout: int = 30) -> str:
    """
    Summarize a question using LiteLLM asynchronously.

    Args:
        question (str): The question to summarize.
        llm_model (str, optional): The LLM model to use, or None for an identity summary (returns the original question).
        logger (logging.Logger, optional): Logger instance for logging execution information.
        api_timeout (int, optional): Timeout for the API call in seconds (default: 30).

    Returns:
        str: The summarized question if successful, or the original question if llm_model is None.

    Raises:
        Exception: If an error occurs during the summarization process.
    """
    _ = load_dotenv(find_dotenv())
    prompt = f"""Summarize the following question into a concise, single sentence that captures its core intent, 
                avoiding greetings, filler words and maintaining clarity: "{question}" """
    try:
        if llm_model:
            response = await acompletion(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                timeout=api_timeout
            )
            summarized_question = response.choices[0].message.content.strip('\"\'')
        else:
            # you can pass None for llm_model for an "Identity" summary i.e. copied directly
            summarized_question = question

        logger.info("Original Question: %s", question)
        logger.info("Summarized Question: %s", f"{summarized_question}\n")
        return summarized_question
    except Exception as e:
        logger.error(f"Error summarizing question: {str(e)}")
        # return question  # Fallback to the original question on error
        # we need to re-throw the error to ensure the task is marked as error
        raise e