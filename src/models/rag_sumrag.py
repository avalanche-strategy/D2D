import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.utils.embedding_utils import summarize_embed_groups, summarize_match_top_k_questions
from src.utils.output_utils import generate_output_from_summarized_matches, setup_logging
from src.utils.api_utils import summarize_question
from sentence_transformers import SentenceTransformer
import torch
from glob import glob
from dotenv import load_dotenv


def main(transcript_dir: str, guidelines_path: str, gpt_model: str, api_key: str, pipeline_name:str, output_path: str,
         disable_logging: bool = False, conciseness: int = 0):
    """
    Main execution function to process transcripts using summarized question embeddings.

    Args:
        transcript_dir (str): Directory containing transcript files.
        guidelines_path (str): Path to the guidelines CSV file.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
        output_path (str): Path for the output CSV file.
    """

    # Set up logging
    logger = setup_logging(pipeline_name, output_path, disable_logging=disable_logging)

    logger.info("Summarizing pipeline started ...")
    logger.info(f"Start finding matches for guidelines: {guidelines_path.split('/')[-1]}")
    logger.info("================================================================================================\n")

    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

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
    for guide_question in guide_questions:
        summarized_guide_question = summarize_question(guide_question, gpt_model, api_key, logger)
        question_embedding = model.encode(summarized_guide_question, convert_to_tensor=True, device=device)
        guide_question_data.append({
            "guide_question": guide_question,
            "summarized_guide_question": summarized_guide_question,
            "embedding": question_embedding
        })
    logger.info("Guide question embeddings precomputed.")
    logger.info("================================================================================================\n")

    matches_list = []

    for transcript_path in transcript_files:

        logger.info(f"Processing transcript: {transcript_path.split('/')[-1]}")
        transcript = load_transcript(transcript_path)

        groups = segment_transcript(transcript)
        group_embeddings = summarize_embed_groups(groups, model, device, gpt_model, api_key, logger)

        transcript_matches = []
        for guide_data in guide_question_data:
            top_k_matches = summarize_match_top_k_questions(guide_data["embedding"], group_embeddings, model, device, gpt_model,
                                                             api_key, k=5, logger=logger)
            transcript_matches.append({
                "guide_question": guide_data["guide_question"],
                "matches": top_k_matches
            })
        matches_list.append(transcript_matches)

    generate_output_from_summarized_matches(transcript_files, matches_list, guide_questions, gpt_model, api_key,
                                            output_path, conciseness=conciseness, logger = logger)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_directory = os.path.join(project_root, "data", "private_data")
    # interview_name = "interview_abcr"
    interview_name = "interview_1090"
    # interview_name = "interview_1003"
    # interview_name = "interview_518"
    # interview_name = "interview_688"
    interviews_directory = os.path.join(data_directory, interview_name)
    guidelines_path = os.path.join(data_directory, f"{interview_name}_guidelines.csv")
    pipeline_name = "rag_sumrag"
    conciseness = 1

    output_path = os.path.join(project_root, "results", f"{pipeline_name}_{interview_name}_conciseness_{conciseness}.csv")
    gpt_model = "gpt-4o-mini"
    main(interviews_directory, guidelines_path, gpt_model, api_key, pipeline_name, output_path,
         disable_logging=False, conciseness=conciseness)
