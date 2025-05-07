import openai

def extract_and_summarize_response_chatgpt(context: str, query: str, gpt_model: str, api_key: str) -> str:
    """
    Query ChatGPT API with the provided context and query.
    
    Args:
        context (str): The dialogue context.
        query (str): The query to answer.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
    
    Returns:
        str: The summarized response from ChatGPT.
    """
    openai.api_key = api_key
    extract_prompt = f"""Given the following dialogue as context: {context}
                        Identify the exact phrase from the interviewee's response 
                        that directly answers the query "{query}" Return only the 
                        core phrase, excluding filler words (e.g., "um," "well") 
                        and irrelevant commentary. If no relevant response is found, 
                        return "[No relevant response found]." """
    try:
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": extract_prompt}
            ]
        )
        extracted_phrase = response.choices[0].message.content
        print("Extract Prompt:", extract_prompt)
        print("\nExtracted Response:", extracted_phrase, "\n")
        summarize_prompt = f"""Given the following extracted response: {extracted_phrase}, 
                            for the query "{query}" Summarize the response into a single, 
                            concise phrase, or just one sentence that captures its core meaning.
                            The number of words of the summary should be less than or equal to 
                            the extracted responses provided. If the response is "[No relevant 
                            response found]," return the same. Do not add introductory phrases 
                            or extra details."""
        try:
            response = openai.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": summarize_prompt}
                ]
            )
            response_content = response.choices[0].message.content.strip('\"\'')
            print("Summarize Prompt:", summarize_prompt)
            print("\nSummarized Response:", response_content, "\n")
            return response_content
        except Exception as e:
            return f"Error querying ChatGPT: {str(e)}"
    except Exception as e:
        return f"Error querying ChatGPT: {str(e)}"