import pytest
import os
import sys
import logging
from unittest.mock import AsyncMock, MagicMock

# Add project root to sys.path for module imports, consistent with test_data_utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.d2d.utils.api_utils import (
    build_extract_prompt,
    build_summarize_prompt,
    extract_and_summarize_response_llm_async,
    summarize_question_async
)


@pytest.fixture
def logger():
    """Provide a logger instance for testing logging behavior."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    return logger


def test_build_extract_prompt_default():
    """
    Test the build_extract_prompt function with the default prompt template.

    Verifies that the default prompt includes the context, query, and expected
    phrases like "[No relevant response found]". Ensures the prompt is correctly
    formatted for LLM extraction tasks.
    """
    # Define sample context and query to simulate a transcript
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"

    # Build the prompt using the default template
    prompt = build_extract_prompt(context, query)

    # Assert that key components of the default prompt are present
    assert "Given the following dialogue as context: " in prompt, "Prompt missing context prefix"
    assert context in prompt, "Prompt does not include context"
    assert f'query "{query}"' in prompt, "Prompt does not include query"
    assert "[No relevant response found]" in prompt, "Prompt missing fallback phrase"


def test_build_extract_prompt_custom():
    """
    Test the build_extract_prompt function with a custom prompt template.

    Ensures that the custom prompt template is correctly formatted with context
    and query placeholders, overriding the default prompt.
    """
    # Define sample context and query
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"

    # Define a custom prompt template with placeholders
    custom_prompt = "Extract from {context} the answer to {query}."

    # Build the prompt using the custom template
    prompt = build_extract_prompt(context, query, custom_prompt)

    # Assert that the custom prompt is formatted correctly
    assert prompt == f"Extract from {context} the answer to {query}.", "Custom prompt incorrectly formatted"


def test_build_summarize_prompt_default():
    """
    Test the build_summarize_prompt function with the default prompt template.

    Verifies that the default prompt includes the extracted phrase, query, and
    summarization instructions, ensuring it’s suitable for LLM summarization.
    """
    # Define sample extracted phrase and query
    extracted_phrase = "I love pizza!"
    query = "What is your favorite food?"

    # Build the prompt using the default template
    prompt = build_summarize_prompt(extracted_phrase, query)

    # Assert that key components of the default prompt are present
    assert "Given the following extracted response: " in prompt, "Prompt missing response prefix"
    assert extracted_phrase in prompt, "Prompt does not include extracted phrase"
    assert f'query "{query}"' in prompt, "Prompt does not include query"
    assert "concise phrase" in prompt, "Prompt missing summarization instruction"


def test_build_summarize_prompt_custom():
    """
    Test the build_summarize_prompt function with a custom prompt template.

    Ensures that the custom prompt template is correctly formatted with extracted
    phrase and query placeholders, overriding the default prompt.
    """
    # Define sample extracted phrase and query
    extracted_phrase = "I love pizza!"
    query = "What is your favorite food?"

    # Define a custom prompt template with placeholders
    custom_prompt = "Summarize {extracted_phrase} for {query}."

    # Build the prompt using the custom template
    prompt = build_summarize_prompt(extracted_phrase, query, custom_prompt)

    # Assert that the custom prompt is formatted correctly
    assert prompt == f"Summarize {extracted_phrase} for {query}.", "Custom prompt incorrectly formatted"


@pytest.mark.asyncio
async def test_extract_and_summarize_response_llm_async_success(mocker, logger):
    """
    Test extract_and_summarize_response_llm_async with successful LLM calls.

    Simulates successful extraction and summarization by mocking LLM responses,
    verifying the returned tuple contains the summarized and extracted phrases.
    """
    # Define sample inputs for the function
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"
    file_name = "test.txt"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to simulate two LLM calls
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())

    # Set up mock response for extraction
    extraction_choice = MagicMock()
    extraction_choice.message = MagicMock()
    extraction_choice.message.content = "I love pizza!"
    extraction_response = MagicMock(choices=[extraction_choice])

    # Set up mock response for summarization
    summarization_choice = MagicMock()
    summarization_choice.message = MagicMock()
    summarization_choice.message.content = "Pizza loved"
    summarization_response = MagicMock(choices=[summarization_choice])

    # Configure the side_effect to return the two responses
    mock_acompletion.side_effect = [extraction_response, summarization_response]

    # Call the async function to extract and summarize
    result = await extract_and_summarize_response_llm_async(
        file_name, context, query, llm_model, logger
    )

    # Assert the result is a tuple with expected summarized and extracted phrases
    assert result == ("Pizza loved", "I love pizza!"), "Incorrect summarized or extracted phrase"

    # Verify that two LLM calls were made (extraction + summarization)
    assert mock_acompletion.call_count == 2, "Expected two LLM calls"

@pytest.mark.asyncio
async def test_extract_and_summarize_response_llm_async_empty_context(mocker, logger):
    """
    Test extract_and_summarize_response_llm_async with an empty context.

    Verifies that an empty context returns "[No relevant response found]" without
    making LLM calls, testing the early exit logic.
    """
    # Define inputs with an empty context
    context = ""
    query = "What is your favorite food?"
    file_name = "test.txt"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to ensure it’s not called
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())

    # Call the async function
    result = await extract_and_summarize_response_llm_async(
        file_name, context, query, llm_model, logger
    )

    # Assert the fallback response is returned
    assert result == "[No relevant response found]", "Expected fallback response for empty context"

    # Verify no LLM calls were made
    assert mock_acompletion.call_count == 0, "LLM should not be called for empty context"


@pytest.mark.asyncio
async def test_extract_and_summarize_response_llm_async_no_relevant_response(mocker, logger):
    """
    Test extract_and_summarize_response_llm_async when no relevant response is found.

    Simulates an LLM extraction returning "[No relevant response found]", verifying
    that the function returns a tuple and skips summarization.
    """
    # Define sample inputs
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"
    file_name = "test.txt"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to return no relevant response
    # Assumes litellm returns choices=[<object with message.content>] structure
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())

    # Set up mock response for extraction
    extraction_choice = MagicMock()
    extraction_choice.message = MagicMock()
    extraction_choice.message.content = "[No relevant response found]"
    extraction_response = MagicMock(choices=[extraction_choice])

    # Configure the mock to return the extraction response
    mock_acompletion.return_value = extraction_response

    # Call the async function
    result = await extract_and_summarize_response_llm_async(
        file_name, context, query, llm_model, logger
    )

    # Assert the result is a tuple with the fallback response for both fields
    assert result == ("[No relevant response found]", "[No relevant response found]"), "Expected fallback tuple"

    # Verify only one LLM call was made (extraction, no summarization)
    assert mock_acompletion.call_count == 1, "Expected only one LLM call"


@pytest.mark.asyncio
async def test_extract_and_summarize_response_llm_async_error(mocker, logger):
    """
    Test extract_and_summarize_response_llm_async with an LLM error.

    Simulates an LLM failure during extraction, verifying that an error message
    is returned and logged appropriately.
    """
    # Define sample inputs
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"
    file_name = "test.txt"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to raise an exception
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())
    mock_acompletion.side_effect = Exception("LLM API error")

    # Call the async function
    result = await extract_and_summarize_response_llm_async(
        file_name, context, query, llm_model, logger
    )

    # Assert the result is an error message
    assert result.startswith("Error querying ChatGPT: LLM API error"), "Expected error message"

    # Verify one LLM call was attempted
    assert mock_acompletion.call_count == 1, "Expected one LLM call attempt"


@pytest.mark.asyncio
async def test_extract_and_summarize_response_llm_async_bad_apikey(mocker, logger, setup_bad_apikey):
    """
    Test extract_and_summarize_response_llm_async with an invalid API key.

    Uses the setup_bad_apikey fixture to simulate an authentication failure,
    verifying that an error message is returned.
    """
    # Define sample inputs
    context = "Interviewer: What's your favorite food?\nInterviewee: I love pizza!"
    query = "What is your favorite food?"
    file_name = "test.txt"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to raise an authentication error
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())
    mock_acompletion.side_effect = Exception("AuthenticationError")

    # Call the async function with bad API key (set by fixture)
    result = await extract_and_summarize_response_llm_async(
        file_name, context, query, llm_model, logger
    )

    # Assert the result is an authentication error message
    assert result.startswith("Error querying ChatGPT: AuthenticationError"), "Expected authentication error"

    # Verify one LLM call was attempted
    assert mock_acompletion.call_count == 1, "Expected one LLM call attempt"


@pytest.mark.asyncio
async def test_summarize_question_async_success(mocker, logger):
    """
    Test summarize_question_async with a successful LLM call.

    Simulates an LLM summarizing a question, verifying that the summarized
    question is returned correctly.
    """
    # Define sample question and LLM model
    question = "What is your favorite food to eat on weekends?"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to return a summarized question
    # Assumes litellm returns a response with choices[0].message.content
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())

    # Set up mock response with the correct structure
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = "Favorite weekend food?"
    mock_response = MagicMock(choices=[choice])

    # Configure the mock to return the mock response when awaited
    mock_acompletion.return_value = mock_response

    # Call the async function
    result = await summarize_question_async(question, llm_model, logger)

    # Assert the summarized question is returned correctly
    assert result == "Favorite weekend food?", "Incorrect summarized question"

    # Verify one LLM call was made
    assert mock_acompletion.call_count == 1, "Expected one LLM call"


@pytest.mark.asyncio
async def test_summarize_question_async_no_model(mocker, logger):
    """
    Test summarize_question_async with no LLM model (identity summary).

    Verifies that passing llm_model=None returns the original question without
    making an LLM call.
    """
    # Define sample question
    question = "What is your favorite food?"
    llm_model = None

    # Mock litellm.acompletion to ensure it’s not called
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())

    # Call the async function
    result = await summarize_question_async(question, llm_model, logger)

    # Assert the original question is returned
    assert result == question, "Expected original question for identity summary"

    # Verify no LLM calls were made
    assert mock_acompletion.call_count == 0, "LLM should not be called with no model"


@pytest.mark.asyncio
async def test_summarize_question_async_error(mocker, logger):
    """
    Test summarize_question_async with an LLM error.

    Simulates an LLM failure, verifying that the function raises an exception
    as expected, rather than falling back to the original question.
    """
    # Define sample question and LLM model
    question = "What is your favorite food?"
    llm_model = "gpt-4o-mini"

    # Mock litellm.acompletion to raise an exception
    mock_acompletion = mocker.patch("src.d2d.utils.api_utils.acompletion", new=AsyncMock())
    mock_acompletion.side_effect = Exception("LLM API error")

    # Assert that the function raises the expected exception
    with pytest.raises(Exception, match="LLM API error"):
        await summarize_question_async(question, llm_model, logger)

    # Verify one LLM call was attempted
    assert mock_acompletion.call_count == 1, "Expected one LLM call attempt"