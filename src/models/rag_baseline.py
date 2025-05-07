import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.utils.embedding_utils import embed_groups, match_top_k_questions
from src.utils.output_utils import generate_output
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from glob import glob
from dotenv import load_dotenv



def main(transcript_dir: str, guidelines_path: str, gpt_model: str, api_key: str, output_path: str):
    """
    Main execution function to process transcripts and generate output.
    
    Args:
        transcript_dir (str): Directory containing transcript files.
        guidelines_path (str): Path to the guidelines CSV file.
        gpt_model (str): The GPT model to use.
        api_key (str): The OpenAI API key.
    """
    print("Program started ...")
    print(f"Start finding matches for guidelines:{guidelines_path}")
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    guide_questions = load_guidelines(guidelines_path)
    transcript_files = glob(os.path.join(transcript_dir, "*.txt"))
    if not transcript_files:
        raise FileNotFoundError("No transcript files found in directory")
    matches_list = []
    for transcript_path in transcript_files:
        transcript = load_transcript(transcript_path)
        groups = segment_transcript(transcript)
        group_embeddings = embed_groups(groups, model, device)
        transcript_matches = []
        for guide_question in guide_questions:
            top_k_matches = match_top_k_questions(guide_question, group_embeddings, model, device, k=5)
            transcript_matches.append({
                "guide_question": guide_question,
                "matches": top_k_matches
            })
        matches_list.append(transcript_matches)
    generate_output(transcript_files, matches_list, guide_questions, gpt_model, api_key, output_path)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_directory = os.path.join(project_root, "data", "private_data")
    interview_name = "interview_1090"
    interviews_directory = os.path.join(data_directory, interview_name)
    guidelines_path = os.path.join(data_directory, f"{interview_name}_guidelines.csv")
    
    output_path = os.path.join(project_root, "results", "rag_baseline_output.csv")
    gpt_model = "gpt-4o-mini"
    main(interviews_directory, guidelines_path, gpt_model, api_key, output_path)