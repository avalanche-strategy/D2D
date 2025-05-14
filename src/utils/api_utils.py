import logging
from litellm import completion
from dotenv import load_dotenv, find_dotenv


def build_extract_prompt(context: str, query: str, conciseness: int = 1) -> str:
    """
    Build the extract prompt for ChatGPT with configurable conciseness.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).

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


def build_summarize_prompt(extracted_phrase: str, query: str, conciseness: int = 1) -> str:
    """
    Build the summarized prompt for ChatGPT with configurable conciseness.

    Args:
        extracted_phrase (str): The phrase extracted from the interviewee's response.
        query (str): The query to answer.
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).

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


def extract_and_summarize_response_chatgpt(context: str, query: str, llm_model: str, api_key: str, conciseness: int = 1,
                                           logger: logging.Logger = None) -> str:
    """
    The core function of the Generator in the pipeline.
    Query LLM with the provided context and query.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        llm_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).
        logger (logging.Logger, optional): Logger instance for logging execution information.
            Defaults to None.

    Returns:
        str: The summarized response from ChatGPT.
    """

    # openai.api_key = api_key
    _ = load_dotenv(find_dotenv())
    # logger.info("Extracting and Summarizing Response ...")

    # First call: Extract the core phrase from the interviewee's response
    try:
        extract_prompt = build_extract_prompt(context, query, conciseness)
        response = completion(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": extract_prompt}
            ],
            temperature=0
        )
        extracted_phrase = response.choices[0].message.content

        # Debugging purpose only
        logger.info("\nExtract Prompt: %s", extract_prompt)
        logger.info("\nExtracted Response: %s", f"{extracted_phrase}\n")

        summarize_prompt = build_summarize_prompt(extracted_phrase, query, conciseness)

        # Second call: Summarize the extracted phrase
        try:
            response = completion(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": summarize_prompt}
                ],
                temperature=0
            )
            response_content = response.choices[0].message.content.strip('\"\'')

            # Debugging purpose only
            logger.info("Summarize Prompt: %s", summarize_prompt)
            logger.info("\nSummarized Response: %s", f"{response_content}\n")

            return response_content
        except Exception as e:
            return f"Error querying ChatGPT: {str(e)}"
    except Exception as e:
        return f"Error querying ChatGPT: {str(e)}"


def summarize_question(question: str, llm_model: str, api_key: str, logger: logging.Logger = None) -> str:
    """
    Summarize a question using the OpenAI API.

    Args:
        question (str): The question to summarize.
        llm_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.

    Returns:
        str: The summarized question.
    """
    _ = load_dotenv(find_dotenv())
    prompt = f"""Summarize the following question into a concise, single sentence that captures its core intent, 
                avoiding greetings, filler words and maintaining clarity: "{question}" """
    try:
        response = completion(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        summarized_question = response.choices[0].message.content.strip('\"\'')
        logger.info("Original Question: %s", question)
        logger.info("Summarized Question: %s", f"{summarized_question}\n")
        return summarized_question
    except Exception as e:
        print(f"Error summarizing question: {str(e)}")
        return question  # Fallback to the original question on error


def extract_and_summarize_response_claude(context: str, query: str, llm_model: str, api_key: str) -> str:
    """
    Query Anthropic API with the provided context and query.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        llm_model (str): The Claude model to use.
        api_key (str): The Anthropic API key.

    Returns:
        str: The response from Anthropic.
    """
    _ = load_dotenv(find_dotenv())

    extract_prompt = f"""Given the following dialogue as context: {context}
                        Identify the exact phrase from the interviewee's response 
                        that directly answers the query "{query}" Return only the 
                        core concise phrases, excluding filler words (e.g., "um," "well") 
                        and irrelevant commentary. If it's possible, please use as 
                        much original wordings as possible. If no relevant response 
                        is found, return "[No relevant response found]." """
    try:
        response = completion(
            model=llm_model,
            max_tokens=100,
            system="You are a market researcher who analyzes interviews to extract correct responses.",
            messages=[
                {
                    "role": "user",
                    "content": extract_prompt
                },
                {
                    "role": "assistant",
                    "content": "Here is the response extracted from the dialogue: <summary>"
                }
            ],
            stop_sequences=["</summary>"]
        )

        print(f"Response: {response.usage}")
        # return response.content[0].text
        extracted_phrase = response.content[0].text
        # print("Extract Prompt:", extract_prompt)
        print("\nExtracted Response:", extracted_phrase, "\n")
        return extracted_phrase

        # summarize_prompt = f"""Given the following extracted response: {extracted_phrase},
        #                     for the query "{query}" Summarize the response into a single,
        #                     concise phrase, or just one sentence that captures its core meaning.
        #                     If it's possible, please use as much original wordings as possible,
        #                     and it doesn't have to be complete sentence. The number of words of
        #                     the summary should be less than or equal to the extracted responses
        #                     provided. If the response is "[No relevant response found]," return
        #                     the same. Do not add introductory phrases or extra details."""
        # try:
        #     response = openai.chat.completions.create(
        #         model=gpt_model,
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": summarize_prompt}
        #         ]
        #     )
        #     response_content = response.choices[0].message.content.strip('\"\'')
        #     print("Summarize Prompt:", summarize_prompt)
        #     print("\nSummarized Response:", response_content, "\n")
        #     return response_content
        # except Exception as e:
        #     return f"Error querying ChatGPT: {str(e)}"
    except Exception as e:
        return f"Error querying Anthropic: {str(e)}"
        logger.info(f"Error summarizing question: {str(e)}")
        return question  # Fallback to the original question on error
