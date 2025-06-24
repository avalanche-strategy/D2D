"""
This module contains functions for building LLM prompt used in the data extraction 
and summarization stages. These prompts are either generated from default templates or 
customized by users when initializing Processor with custom_extract_prompt and custom_summarize_prompt.
"""

def build_extract_prompt(context: str, query: str, custom_extract_prompt: str = None) -> str:
    """
    Build the extract prompt for LLM.

    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        custom_extract_prompt (str, optional): Custom prompt template with {context} and {query} placeholders.

    Returns:
        str: The formatted extract prompt.
    """
    if custom_extract_prompt:
        # Use the custom prompt template, replacing placeholders
        return custom_extract_prompt.format(context=context, query=query)

    # Balanced: Concise but understandable (default)
    return f"""Given the following dialogue as context: {context}
                         When referring to a person, use their name explicitly and avoid pronouns 
                                   like "he" or "her". Identify a concise phrase from the interviewee's 
                                   response that directly answers the query "{query}", ensuring the phrase is 
                                   understandable with sufficient context. Exclude filler words (e.g., "um," 
                                   "well") and irrelevant commentary, but retain enough original wording 
                                   to maintain clarity. If no relevant response is found, return 
                                   "[No relevant response found]."  """


def build_summarize_prompt(extracted_phrase: str, query: str, custom_summarize_prompt: str = None) -> str:
    """
    Build the summarized prompt for LLM.

    Args:
        extracted_phrase (str): The phrase extracted from the interviewee's response.
        query (str): The query to answer.
        custom_summarize_prompt (str, optional): Custom prompt template with {extracted_phrase} and {query} placeholders.

    Returns:
        str: The formatted summarized prompt.
    """
    if custom_summarize_prompt:
        # Use the custom prompt template, replacing placeholders
        return custom_summarize_prompt.format(extracted_phrase=extracted_phrase, query=query)

    # Balanced: Concise but understandable (default)
    return f"""Given the following extracted response: {extracted_phrase}, 
                        for the query "{query}" When referring to a third party, use their name 
                        explicitly and avoid pronouns like "he/she", "him/her" or "it/they". Summarize the response 
                        into a single, concise phrase that directly answers the query, focusing 
                        exclusively on the primary reason or sentiment expressed. Use only the words 
                        present in the extracted response, without rephrasing or adding new words. 
                        Remove unnecessary words, excluding secondary details entirely, to ensure the 
                        phrase retains the key verb and subject and maintains semantic clarity. The 
                        result must be a coherent phrase, not broken into disconnected fragments. It 
                        doesn’t have to be a complete sentence. Do not add introductory phrases or extra 
                        details."""
