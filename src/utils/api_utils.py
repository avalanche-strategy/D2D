import openai
import anthropic

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
                        and irrelevant commentary. If it's possible, please use as 
                        much original wordings as possible. If no relevant response 
                        is found, return "[No relevant response found]." """
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
                            If it's possible, please use as much original wordings as possible, 
                            and it doesn't have to be complete sentence. The number of words of 
                            the summary should be less than or equal to the extracted responses 
                            provided. If the response is "[No relevant response found]," return 
                            the same. 
                            
                            If there are multiple elements, separate them with comma. Do not return a bullet list.

                            Do not return any preamble or explanations, return only an unquoted string response.                            
                            """
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


def summarize_question(question: str, gpt_model: str, api_key: str) -> str:
    """
    Summarize a question using the OpenAI API.

    Args:
        question (str): The question to summarize.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.

    Returns:
        str: The summarized question.
    """
    openai.api_key = api_key
    prompt = f"""Summarize the following question into a concise, single sentence that captures its core intent, 
                avoiding greetings, filler words and maintaining clarity: "{question}" """
    try:
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        summarized_question = response.choices[0].message.content.strip('\"\'')
        print("Original Question:", question)
        print("Summarized Question:", summarized_question, "\n")
        return summarized_question
    except Exception as e:
        print(f"Error summarizing question: {str(e)}")
        return question  # Fallback to original question on error
    

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
    client = anthropic.Anthropic(api_key=api_key)

    extract_prompt = f"""Given the following dialogue as context: {context}
                        Identify the exact phrase from the interviewee's response 
                        that directly answers the query "{query}" Return only the 
                        core concise phrases, excluding filler words (e.g., "um," "well") 
                        and irrelevant commentary. If it's possible, please use as 
                        much original wordings as possible. If no relevant response 
                        is found, return "[No relevant response found]." """
    try:
        response = client.messages.create(
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
        #return response.content[0].text
        extracted_phrase = response.content[0].text
        #print("Extract Prompt:", extract_prompt)
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