import sys
import os
import time

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.d2d import D2DProcessor


def main_top_k():
    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4.1-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_K,
    )

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "private_data")
    interview_name = "interview_1090"
    output_dir = os.path.join(root_dir, "results")

    # Step 3: Start transcripts processing
    processor.process_transcripts(
        data_dir=data_dir,
        interview_name=interview_name,
        output_dir=output_dir,
        disable_logging=False
    )
    # Process completed


def main_top_p():
    # Step 1: Initialize the processor
    processor = D2DProcessor(
        llm_model="gpt-4.1-mini",
        embedding_model="multi-qa-mpnet-base-dot-v1",
        max_concurrent_calls=10,
        sampling_method=D2DProcessor.SamplingMethod.TOP_P,
        top_p=0.5,
    )

    # Step 2: Define paths relative to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(root_dir, "data", "private_data")
    interview_name = "interview_1090"
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

    main_top_p()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")