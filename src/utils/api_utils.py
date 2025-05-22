import logging
from litellm import completion, acompletion
from dotenv import load_dotenv, find_dotenv


def build_extract_prompt(context: str, query: str) -> str:
    """
    Build the extract prompt for LLM.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.

    Returns:
        str: The formatted extract prompt.
    """
    # Balanced: Concise but understandable
    return f"""Given the following dialogue as context: {context}
                         When referring to a person, use their name explicitly and avoid pronouns 
                                   like "he" or "her". Identify a concise phrase from the interviewee's 
                                   response that directly answers the query "{query}", ensuring the phrase is 
                                   understandable with sufficient context. Exclude filler words (e.g., "um," 
                                   "well") and irrelevant commentary, but retain enough original wording 
                                   to maintain clarity. If no relevant response is found, return 
                                   "[No relevant response found]."  """


def build_summarize_prompt(extracted_phrase: str, query: str) -> str:
    """
    Build the summarized prompt for LLM.

    Args:
        extracted_phrase (str): The phrase extracted from the interviewee's response.
        query (str): The query to answer.

    Returns:
        str: The formatted summarized prompt.
    """
    # Balanced: Concise but understandable
    return f"""Given the following extracted response: {extracted_phrase}, 
                        for the query "{query}" When referring to a person, use their name 
                        explicitly and avoid pronouns like "he/she", "him/her" or "it/they". Summarize the response 
                        into a single, concise phrase that directly answers the query, focusing 
                        exclusively on the primary reason or sentiment expressed. Use only the words 
                        present in the extracted response, without rephrasing or adding new words. 
                        Remove unnecessary words, excluding secondary details entirely, to ensure the 
                        phrase retains the key verb and subject and maintains semantic clarity. The 
                        result must be a coherent phrase, not broken into disconnected fragments. It 
                        doesn't have to be a complete sentence. If the response is "[No relevant 
                        response found]," return the same. Do not add introductory phrases or extra 
                        details."""

async def extract_and_summarize_response_llm_async(context: str, query: str, llm_model: str,
                                                logger: logging.Logger = None) -> str:
    """
    The core function of the Generator in the pipeline.
    Query LLM with the provided context and query asynchronously.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        llm_model (str): The LLM model to use.
        api_key (str): The OpenAI API key.
        logger (logging.Logger, optional): Logger instance for logging execution information.

    Returns:
        str: The summarized response from ChatGPT.
    """
    _ = load_dotenv(find_dotenv())

    # First call: Extract the core phrase from the interviewee's response
    try:
        if len(context.strip())==0:
            # no matches using p
            logger.info("Line Referencing: No matching dialog, appending [No relevant response found]")
            # this could be a different string, if we need to debug/differentiate
            return "[No relevant response found]"
        
        extract_prompt = build_extract_prompt(context, query)
        response = await acompletion(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": extract_prompt}
            ],
            temperature=0
        )
        extracted_phrase = response.choices[0].message.content
        summarize_prompt = build_summarize_prompt(extracted_phrase, query)

        # Second call: Summarize the extracted phrase
        try:
            response = await acompletion(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": summarize_prompt}
                ],
                temperature=0
            )
            response_content = response.choices[0].message.content.strip('\"\'')
            return (response_content, extracted_phrase)
        except Exception as e:
            logger.error(f"Error summarizing response: {str(e)}")
            return f"Error querying ChatGPT: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting response: {str(e)}")
        return f"Error querying ChatGPT: {str(e)}"

async def summarize_question_async(question: str, llm_model: str, logger: logging.Logger = None) -> str:
    """
    Summarize a question using the LiteLLM asynchronously.

    Args:
        question (str): The question to summarize.
        llm_model (str): The GPT model to use.
        logger (logging.Logger, optional): Logger instance for logging execution information.

    Returns:
        str: The summarized question.
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
                ]
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
        return question  # Fallback to the original question on error

