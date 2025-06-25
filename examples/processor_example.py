import sys
import os
import time
from unittest.mock import patch, Mock
from litellm.exceptions import AuthenticationError, Timeout, APIError

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, "data", "synthetic_data")
sys.path.append(root_dir)

from d2d import D2DProcessor

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


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process completed in {elapsed_time:.4f} seconds")