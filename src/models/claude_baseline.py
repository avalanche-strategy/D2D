import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.data_utils import load_guidelines, load_transcript, segment_transcript
from src.utils.embedding_utils import embed_groups, match_top_p_questions
from src.utils.output_utils import generate_claude_output
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from glob import glob
from dotenv import load_dotenv



def main(transcript_dir: str, guidelines_path: str, output_path: str, api_key: str, 
         embedding_model: str = "sentence-transformers/nli-roberta-base-v2", 
         llm_model: str = "claude-3-5-haiku-20241022", 
         ):
    """
    Main execution function to process transcripts and generate output.
    
    Args:
        transcript_dir (str): Directory containing transcript files.
        guidelines_path (str): Path to the guidelines CSV file.
        output_path (str): Directory where output CSV will be saved.
        embedding_model (str): SentenceTransformer to use to embed Guidelines+Transcript. Default is nli-roberta-base-v2
        llm_model (str): The Anthropic model to to use. Default is claude-3-5-haiku-20241022
        api_key (str): The Anthropic API key.
    """
    print("Program started ...")
    print(f"Start finding matches for guidelines:{guidelines_path}")
    # the transformer prints a warning without this setting
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer(embedding_model)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    guide_question_list = load_guidelines(guidelines_path)
    # TODO: At this point, clean the questions or not?
    #temp = glob(os.path.join(transcript_dir, "*.txt"))
    #transcript_file_list = temp[:1]
    transcript_file_list = glob(os.path.join(transcript_dir, "*.txt"))

    if not transcript_file_list:
        raise FileNotFoundError("No transcript files found in directory")
    matches_list = []
    for transcript_file in transcript_file_list:
        transcript = load_transcript(transcript_file)
        groups = segment_transcript(transcript)
        # embed each group
        group_embeddings = embed_groups(groups, model, device)
        print(f"File: {transcript_file}")
        print()
        # for g in group_embeddings:
        #     print(g['embedding'][:5])
        #     print('Interviewer: ', g['interviewer_question'])
        #     print('Interviewee: ', g['interviewee_response'], end=f"\n{'-'*20}\n\n")
        # print("*"*30)     
        # TODO: At this point, many choices! 
        # (1) Embed Interviewer + Interviewee or just Interviewer? This will be basis for matching to guidelines
        # (2) Clean up the Interviewer part to just extract the question only?
        transcript_matches = []
        for guide_question in guide_question_list:
            top_k_matches = match_top_p_questions(guide_question, group_embeddings, model, device, p=0.6, max_matches=5)
            transcript_matches.append({
                "guide_question": guide_question,
                "matches": top_k_matches
            })
        for tm in transcript_matches:
            print(f"Guide Question: ", tm["guide_question"])
            for top_p in tm["matches"]:
                print(f"{top_p['similarity']:.4f}")
                print(f"{top_p['speaking_round']}")
                print('Interviewer: ', top_p['question'])
                print('Interviewee: ', top_p['response'], end=f"\n{'-'*20}\n\n")
            print("*"*30)  
            print() 

        matches_list.append(transcript_matches)
        
    #generate_claude_output(transcript_file_list, matches_list, guide_questions, llm_model, api_key, output_path)
    generate_claude_output(transcript_file_list, matches_list, guide_question_list, llm_model, api_key, output_path)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_directory = os.path.join(project_root, "data", "private_data")
    interview_name = "interview_1090"
    interviews_directory = os.path.join(data_directory, interview_name)
    guidelines_path = os.path.join(data_directory, f"{interview_name}_guidelines.csv")
    
    output_path = os.path.join(project_root, "results", "claude_baseline.csv")
    # call with default embedding and llm model
    main(interviews_directory, guidelines_path, api_key=api_key, output_path=output_path)