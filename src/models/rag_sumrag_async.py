import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.utils.embedding_utils import summarize_embed_groups_async, summarize_match_top_k_questions_async
from src.utils.output_utils import generate_output_from_summarized_matches_async, setup_logging, output_divider
from src.utils.api_utils import summarize_question_async
from sentence_transformers import SentenceTransformer
import torch
from glob import glob
from dotenv import load_dotenv
import time

async def main(transcript_dir: str, guidelines_path: str, llm_model: str, embedding_model:str, pipeline_name: str, output_path: str,
               disable_logging: bool = False, max_concurrent_calls: int = 10):
    """
    Main execution function to process transcripts using summarized question embeddings.

    Args:
        transcript_dir (str): Directory containing transcript files.
        guidelines_path (str): Path to the guidelines CSV file.
        llm_model (str): The GPT model to use.
        output_path (str): Path for the output CSV file.
        disable_logging (bool): Whether to disable logging.
    """
    # Set up logging
    logger = setup_logging(pipeline_name, output_path, disable_logging=disable_logging)

    logger.info("Summarizing pipeline started ...")
    logger.info(f"Start finding matches for guidelines: {guidelines_path.split('/')[-1]}")
    output_divider(logger, True)

    embedding_model = SentenceTransformer(embedding_model)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    guide_questions = load_guidelines(guidelines_path)

    transcript_files = glob(os.path.join(transcript_dir, "*.txt"))
    if not transcript_files:
        raise FileNotFoundError("No transcript files found in directory")

    # Precompute embeddings for guide questions
    logger.info("Summarizing and embedding guide questions...")
    guide_question_data = []
    semaphore = asyncio.Semaphore(max_concurrent_calls)  # Limit to 10 concurrent API calls

    async def summarize_with_limit(question):
        async with semaphore:
            return await summarize_question_async(question, llm_model, logger)

    tasks = [summarize_with_limit(guide_question) for guide_question in guide_questions]
    summarized_questions = await asyncio.gather(*tasks)

    for guide_question, summarized_guide_question in zip(guide_questions, summarized_questions):
        question_embedding = embedding_model.encode(summarized_guide_question, convert_to_tensor=True, device=device)
        guide_question_data.append({
            "guide_question": guide_question,
            "summarized_guide_question": summarized_guide_question,
            "embedding": question_embedding
        })

    logger.info("Guide question embeddings precomputed.")
    output_divider(logger, True)

    matches_list = []

    for transcript_path in transcript_files:
        logger.info(f"Processing transcript: {transcript_path.split('/')[-1]}")
        transcript = load_transcript(transcript_path)

        groups = segment_transcript(transcript)
        group_embeddings = await summarize_embed_groups_async(groups, embedding_model, device, llm_model, logger)

        transcript_matches = []

        for guide_data in guide_question_data:
            top_k_matches = await summarize_match_top_k_questions_async(guide_data["embedding"], group_embeddings, k=5)
            transcript_matches.append({
                "guide_question": guide_data["guide_question"],
                "matches": top_k_matches
            })
        matches_list.append(transcript_matches)

    await generate_output_from_summarized_matches_async(
        transcript_files, matches_list, guide_questions, llm_model, output_path, logger=logger,
        embedding_model=embedding_model, device=device, max_concurrent_calls=max_concurrent_calls
    )


if __name__ == "__main__":
    load_dotenv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_directory = os.path.join(project_root, "data", "private_data")
    interview_name = "interview_1090"
    embedding_model = "multi-qa-mpnet-base-dot-v1"
    # interview_name = "interview_abcr"
    interviews_directory = os.path.join(data_directory, interview_name)
    guidelines_path = os.path.join(data_directory, f"{interview_name}_guidelines.csv")
    pipeline_name = "rag_sumrag_async"

    output_path = os.path.join(project_root, "results", f"{pipeline_name}_{interview_name}.csv")
    llm_model = "gpt-4.1-mini"

    start_time = time.time()

    # Run the async main function
    asyncio.run(main(
        interviews_directory, guidelines_path, llm_model, embedding_model, pipeline_name, output_path,
        disable_logging=False
    ))

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")