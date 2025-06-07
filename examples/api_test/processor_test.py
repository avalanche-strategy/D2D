import sys
import os
import time

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.d2d import D2DProcessor

"""
For more information about the D2DProcessor parameters, refer to D2D/docs/example.ipynb. 
This file includes several common scenarios. The scenarios are explained in detail below.
"""

# Example 1 - Customizing Top-k sampling method.
def test_top_k():
    """
    Tests the D2DProcessor with the TOP_K sampling method.
    This scenario initializes the processor with a specific language model, embedding model,
    and the TOP_K sampling strategy to process interview transcripts. It verifies that the
    processor can handle transcript processing with deterministic top-k sampling, which selects
    the top k most likely tokens during text generation. The test uses synthetic interview data
    and saves results to the output directory.
    """
    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
    )

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "synthetic_data")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )
    # Process completed

# Example 2 - Customizing Top-p sampling method.
def test_top_p():
    """
    Tests the D2DProcessor with the TOP_P (nucleus) sampling method.
    This scenario configures the processor with TOP_P sampling, which selects tokens based on
    a cumulative probability threshold (top_p=0.5). It tests the processor's ability to process
    interview transcripts with a probabilistic sampling approach, allowing for more diverse outputs
    compared to TOP_K. The test uses synthetic interview data and saves results to the output directory.
    """
    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_P,
        top_p=0.5,
    )

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "synthetic_data")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )
    # Process completed

# Example 3 - Customizing the extraction and summarization prompts with custom settings.
def test_custom_prompt():
    """
    Tests the D2DProcessor with custom extraction and summarization prompts.
    This scenario demonstrates how to use optional custom prompts to tailor the processor's
    extraction and summarization behavior. The custom prompts are designed to extract a short
    phrase from the interviewee and summarize it based on a query. The test uses TOP_K sampling
    and synthetic interview data. Note that the provided prompts are for demonstration only and
    may produce suboptimal results; they should be adjusted for specific use cases.
    """
    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
        custom_extract_prompt="""Using the dialogue: {context}, find a short phrase from the interviewee that answers '{query}'. Avoid pronouns and use explicit names. If no answer is found, return '[No answer found]'.""",
        custom_summarize_prompt="""From the phrase: {extracted_phrase}, for the query '{query}', create a brief summary using only the original words, focusing on the main point."""
    )

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "synthetic_data")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )
    # Process completed

# Example 4 - Minimal processor initialization with default parameters.
def test_minimal_init():
    """
    Tests the D2DProcessor with a minimal configuration.
    This scenario initializes the processor with default parameters, omitting custom settings
    such as sampling method, custom prompts, or specific models. It verifies that the processor
    can function correctly with minimal setup, relying on default values for processing interview
    transcripts. The test uses synthetic interview data and saves results to the output directory.
    """
    # Step 1: Initialize the processor with no custom parameter
    processor = D2DProcessor()

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "synthetic_data")
    interview_name = "interview_food"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )
    # Process completed

# Example 5 - Thematic alignment mismatch scenario.
def test_thematic_alignment_mismatch_transcript():
    """
    Tests the D2DProcessor's thematic alignment check with a mismatched transcript.
    This scenario processes a transcript ('movie_001.txt') in the directory, D2D/data/interview_foodMismatch,
    that significantly deviates from the guideline questions. It verifies that the processor
    detects low similarity between the transcript and guideline questions, triggers a user prompt
    to confirm continuation, and skips processing if the user enters 'n'. The test
    uses a minimal processor configuration and synthetic data to simulate the mismatch scenario.
    Note: The time taken for this test is significantly longer than other examples because it requires user input.
    """
    # Step 1: Initialize the processor with no custom parameter
    processor = D2DProcessor()

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "synthetic_data")
    interview_name = "interview_foodMismatch"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging_to_console=True
    )
    # Process completed


if __name__ == "__main__":
    start_time = time.time()

    # test_top_p()
    # test_custom_prompt()
    # test_top_k()
    # test_minimal_init()
    test_thematic_alignment_mismatch_transcript()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Process completed in {elapsed_time:.4f} seconds")