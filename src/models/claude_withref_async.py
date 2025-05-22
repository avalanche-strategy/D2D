import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.utils.embedding_utils import embed_groups, match_top_p_questions
from src.utils.output_utils import generate_output_from_summarized_matches_async, setup_logging, output_divider
from src.utils.api_utils import summarize_question_async
from sentence_transformers import SentenceTransformer
import torch
from glob import glob
from dotenv import load_dotenv
import time

async def main(transcript_dir: str, guidelines_path: str, llm_model: str, pipeline_name: str, output_path: str,
               disable_logging: bool = False, conciseness: int = 0):
    """
    Main execution function to process transcripts using summarized question embeddings.

    Args:
        transcript_dir (str): Directory containing transcript files.
        guidelines_path (str): Path to the guidelines CSV file.
        llm_model (str): The GPT model to use.
        output_path (str): Path for the output CSV file.
        disable_logging (bool): Whether to disable logging.
        conciseness (int): Conciseness level (0 for less concise, 1 for more concise).
    """
    # Set up logging
    logger = setup_logging(pipeline_name, output_path, disable_logging=disable_logging)

    logger.info("Summarizing pipeline started ...")
    logger.info(f"Start finding matches for guidelines: {guidelines_path.split('/')[-1]}")
    output_divider(logger, True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer("sentence-transformers/nli-roberta-base-v2", device=device)

    print(guidelines_path)
    guide_questions = load_guidelines(guidelines_path)

    transcript_files = glob(os.path.join(transcript_dir, "*.txt"))
    if not transcript_files:
        raise FileNotFoundError("No transcript files found in directory")

    # ===============================================================================================================
    # Precompute embeddings for guide questions
    logger.info("Summarizing and embedding guide questions...")
    guide_question_data = []
    semaphore = asyncio.Semaphore(1)  # Limit to 10 concurrent API calls

    async def summarize_with_limit(question):
        async with semaphore:
            return await summarize_question_async(question, None, logger)

    tasks = [summarize_with_limit(guide_question) for guide_question in guide_questions]
    summarized_questions = await asyncio.gather(*tasks)

    for guide_question, summarized_guide_question in zip(guide_questions, summarized_questions):
        question_embedding = model.encode(summarized_guide_question, convert_to_tensor=True, device=device)
        guide_question_data.append({
            "guide_question": guide_question,
            "summarized_guide_question": summarized_guide_question,
            "embedding": question_embedding
        })

    logger.info("Guide question embeddings precomputed.")
    output_divider(logger, True)
    # ===============================================================================================================

    # ===============================================================================================================
    # Looking for matches to questions
    matches_list = []
    for transcript_path in transcript_files:
        logger.info(f"Processing transcript: {transcript_path.split('/')[-1]}")
        transcript = load_transcript(transcript_path)

        groups = segment_transcript(transcript)
        group_embeddings = embed_groups(groups, model, device)

        transcript_matches = []
        for guide_data in guide_question_data:
            top_matches = match_top_p_questions(guide_data["embedding"], group_embeddings, p=0.5)
            transcript_matches.append({
                "guide_question": guide_data["guide_question"],
                "matches": top_matches
            })
        matches_list.append(transcript_matches)
    # ===============================================================================================================

    # ===============================================================================================================
    await generate_output_from_summarized_matches_async(
        transcript_files, matches_list, guide_questions, llm_model, output_path, conciseness=conciseness, logger=logger,
        embedding_model=model, device=device
    )
    # ===============================================================================================================


if __name__ == "__main__":
    load_dotenv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_directory = os.path.join(project_root, "data", "private_data")
    
    interview_name = "interview_1090"
    interviews_directory = os.path.join(data_directory, interview_name)
    guidelines_path = os.path.join(data_directory, f"{interview_name}_guidelines.csv")
    pipeline_name = "claude_withref_async"
    conciseness = 1

    output_path = os.path.join(project_root, "results", f"{pipeline_name}_{interview_name}.csv")
    llm_model = "claude-3-5-haiku-20241022"

    start_time = time.time()

    # Run the async main function
    asyncio.run(main(
        interviews_directory, guidelines_path, llm_model, pipeline_name, output_path,
        disable_logging=False, conciseness=conciseness
    ))

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")