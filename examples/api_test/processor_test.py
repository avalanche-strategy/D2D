import sys
import os
import time

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.d2d import D2DProcessor

# Example 1
def main_top_k():
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
        disable_logging=False
    )
    # Process completed

# Example 2
def main_top_p():
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
        disable_logging=False
    )
    # Process completed

# Example 3
def main_custom_prompt():
    """
     Example usage of custom prompts and summarization
     Note that
     1. The custom prompts and summarization are optional and can be omitted if not needed.
     2. These prompts are just for demonstration purposes.
        - The output of these custom prompts is not good.
        - They should be adjusted to fit your specific use case.
    """


    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4o-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
        custom_extract_prompt="""Using the dialogue: {context}, find a short phrase from the interviewee that answers '{query}'. Avoid pronouns and use explicit names. If no answer is found, return '[No answer found]'.""",
        custom_summarize_prompt = """From the phrase: {extracted_phrase}, for the query '{query}', create a brief summary using only the original words, focusing on the main point."""
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
        disable_logging=False
    )
    # Process completed

# Example 4
def main_minimal_init():
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
        disable_logging=False
    )
    # Process completed




if __name__ == "__main__":
    start_time = time.time()

    # main_top_p()
    # main_custom_prompt()
    # main_top_k()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")