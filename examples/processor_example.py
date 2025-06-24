import sys
import os
import time
from unittest.mock import patch, Mock
from litellm.exceptions import AuthenticationError, Timeout, APIError

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, "data", "synthetic_data")
sys.path.append(root_dir)

from src.d2d import D2DProcessor

"""
For more information about the D2DProcessor parameters, refer to D2D/docs/example.ipynb. 
This file includes several common scenarios and error demos.
"""

# --- General Usage Examples ---

def test_top_k():
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
    )
    transcript_dir = os.path.join(data_dir, "transcripts_food")
    guidelines_path = os.path.join(data_dir, "interview_food_guidelines.csv")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")
    processor.process_transcripts(
        transcripts_dir=transcript_dir,
        guidelines_path=guidelines_path,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )

def test_top_p():
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_P,
        top_p=0.5,
    )
    transcript_dir = os.path.join(data_dir, "transcripts_food")
    guidelines_path = os.path.join(data_dir, "interview_food_guidelines.csv")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")
    processor.process_transcripts(
        transcripts_dir=transcript_dir,
        guidelines_path=guidelines_path,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )

def test_custom_prompt():
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
        custom_extract_prompt="""Using the dialogue: {context}, find a short phrase from the interviewee that answers '{query}'. Avoid pronouns and use explicit names. If no answer is found, return '[No answer found]'.""",
        custom_summarize_prompt="""From the phrase: {extracted_phrase}, for the query '{query}', create a brief summary using only the original words, focusing on the main point."""
    )
    transcript_dir = os.path.join(data_dir, "transcripts_food")
    guidelines_path = os.path.join(data_dir, "interview_food_guidelines.csv")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")
    processor.process_transcripts(
        transcripts_dir=transcript_dir,
        guidelines_path=guidelines_path,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )

def test_minimal_init():
    processor = D2DProcessor()
    transcript_dir = os.path.join(data_dir, "transcripts_food")
    guidelines_path = os.path.join(data_dir, "interview_food_guidelines.csv")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")
    processor.process_transcripts(
        transcripts_dir=transcript_dir,
        guidelines_path=guidelines_path,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )

def test_thematic_alignment_mismatch_transcript():
    processor = D2DProcessor()
    transcript_dir = os.path.join(data_dir, "transcripts_foodMismatch")
    guidelines_path = os.path.join(data_dir, "interview_foodMismatch_guidelines.csv")
    interview_name = "interview_foodMismatch"
    output_dir = os.path.join(root_dir, "results")
    processor.process_transcripts(
        transcripts_dir=transcript_dir,
        guidelines_path=guidelines_path,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )

# --- LLM Connection Error Demonstrations ---

def demo_llm_authentication_error():
    """Simulate API authentication failure during LLM connection test."""
    print("Running demo_llm_authentication_error()... This may take some time to complete.")

    def mock_completion(*args, **kwargs):
        raise AuthenticationError("Mocked API key error.", llm_provider="openai", model="gpt-4o-mini")

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing authentication error...")
        try:
            processor._test_llm_connection()
        except Exception as e:
            print(f"Caught exception as expected: {e}")

def demo_llm_timeout_error():
    """Simulate LLM API timeout."""
    print("Running demo_llm_timeout_error()... This may take some time to complete.")

    def mock_completion(*args, **kwargs):
        raise Timeout("Mocked timeout.", llm_provider="openai", model="gpt-4o-mini")

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing timeout error...")
        try:
            processor._test_llm_connection()
        except Exception as e:
            print(f"Caught exception as expected: {e}")

def demo_llm_service_unavailable():
    """Simulate LLM service unavailable (503 error)."""
    print("Running demo_llm_service_unavailable()... This may take some time to complete.")

    def mock_completion(*args, **kwargs):
        raise APIError(message="Mocked service unavailable error.", status_code=503, llm_provider="openai", model="gpt-4o-mini")

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing service unavailable error (503)...")
        try:
            processor._test_llm_connection()
        except Exception as e:
            print(f"Caught exception as expected: {e}")

def demo_llm_rate_limit():
    """Simulate LLM API rate limiting (429 error)."""
    print("Running demo_llm_rate_limit()... This may take some time to complete.")

    def mock_completion(*args, **kwargs):
        error = APIError(message="Mocked rate limit error.", status_code=429, llm_provider="openai", model="gpt-4o-mini")
        raise error

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing rate limit error (429)...")
        try:
            processor._test_llm_connection()
        except Exception as e:
            print(f"Caught exception as expected: {e}")

def demo_llm_unexpected_error():
    """Simulate unexpected exception during LLM connection test."""
    print("Running demo_llm_unexpected_error()... This may take some time to complete.")

    def mock_completion(*args, **kwargs):
        raise Exception("Mocked unexpected error.")

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing unexpected error...")
        try:
            processor._test_llm_connection()
        except Exception as e:
            print(f"Caught exception as expected: {e}")

def demo_llm_model_switch():
    """
    Demonstrate model switching: first three attempts fail (simulate 503), fallback model then succeeds.
    """
    print("Running demo_llm_model_switch()... This may take some time to complete.")

    call_count = {'count': 0}

    def mock_completion(*args, **kwargs):
        if call_count['count'] < 3:
            call_count['count'] += 1
            raise APIError(message="Mocked service unavailable.", status_code=503, llm_provider="openai", model="gpt-4o-mini")
        else:
            # Return a mock completion object as OpenAI would
            return Mock(choices=[Mock(message=Mock(content="Success from fallback"))])

    with patch("src.d2d.processor.completion", mock_completion):
        processor = D2DProcessor(llm_model="gpt-4o-mini")
        print("Testing model switch with fallback...")
        result = processor._test_llm_connection()
        print(f"Model switch result: {result}")
        print(f"Final model in processor: {processor.llm_model}")

# --- Main ---

if __name__ == "__main__":
    start_time = time.time()

    """
    Uncomment any test/demo you want to run
    Please run each test/demo individually to avoid interference with other tests/demos
    """

    # --- General Usage Examples ---
    # test_top_k()
    # test_top_p()
    # test_custom_prompt()
    test_minimal_init()
    # test_thematic_alignment_mismatch_transcript()

    # --- LLM Connection error demos ---
    # demo_llm_authentication_error()
    # demo_llm_timeout_error()
    # demo_llm_service_unavailable()
    # demo_llm_rate_limit()
    # demo_llm_unexpected_error()
    # demo_llm_model_switch()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process completed in {elapsed_time:.4f} seconds")