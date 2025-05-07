from sentence_transformers import SentenceTransformer, util
from src.utils.api_utils import summarize_question
import torch

def embed_groups(groups: list[dict], model: SentenceTransformer, device: torch.device) -> list[dict]:
    """
    Embed interviewer questions from the groups.
    
    Args:
        groups (list[dict]): List of groups containing interviewer and interviewee turns.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
    
    Returns:
        list[dict]: List of dictionaries with embeddings and original texts.
    """
    group_embeddings = []
    for group in groups:
        interviewer_question = " ".join(group["interviewer"]).replace("Interviewer: ", "")
        embedding = model.encode(interviewer_question, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "interviewer_question": interviewer_question,
            "interviewee_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding
        })
    return group_embeddings

def match_top_k_questions(guide_question: str, group_embeddings: list[dict], model: SentenceTransformer, device: torch.device, k: int = 3) -> list[dict]:
    """
    Match a guideline question to the top k groups based on similarity.
    
    Args:
        guide_question (str): The guide question to match.
        group_embeddings (list[dict]): List of embedded groups.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        k (int): Number of top matches to return.
    
    Returns:
        list[dict]: Top k matches with similarity scores.
    """
    question_embedding = model.encode(guide_question, convert_to_tensor=True, device=device)
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(question_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["interviewee_response"],
            "question": group["interviewer_question"],
            "similarity": float(similarity)
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


def summarize_embed_groups(groups: list[dict], model: SentenceTransformer, device: torch.device, gpt_model: str,
                            api_key: str) -> list[dict]:
    """
    Embed summarized interviewer questions from the groups.

    Args:
        groups (list[dict]): List of groups containing interviewer and interviewee turns.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        gpt_model (str): The GPT model for summarization.
        api_key (str): The OpenAI API key for summarization.

    Returns:
        list[dict]: List of dictionaries with embeddings, summarized questions, and original texts.
    """
    group_embeddings = []
    for group in groups:
        original_question = " ".join(group["interviewer"]).replace("Interviewer: ", "")
        summarized_question = summarize_question(original_question, gpt_model, api_key)
        embedding = model.encode(summarized_question, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "original_question": original_question,
            "summarized_question": summarized_question,
            "original_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding
        })
    return group_embeddings


def summarize_match_top_k_questions(guide_question: str, group_embeddings: list[dict], model: SentenceTransformer,
                                     device: torch.device, gpt_model: str, api_key: str, k: int = 3) -> list[dict]:
    """
    Match a summarized guideline question to the top k groups based on similarity.

    Args:
        guide_question (str): The guide question to match.
        group_embeddings (list[dict]): List of embedded groups.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        gpt_model (str): The GPT model for summarization.
        api_key (str): The OpenAI API key for summarization.
        k (int): Number of top matches to return.

    Returns:
        list[dict]: Top k matches with similarity scores, using original texts.
    """
    summarized_guide_question = summarize_question(guide_question, gpt_model, api_key)
    print(f"Original Question: {guide_question}")
    print(f"Summarized Question: {summarized_guide_question}")
    question_embedding = model.encode(summarized_guide_question, convert_to_tensor=True, device=device)
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(question_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["original_response"],
            "question": group["original_question"],
            "similarity": float(similarity)
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]