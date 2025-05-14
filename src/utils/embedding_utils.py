import logging

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
    # adding index so we know the order of speaking
    for index, group in enumerate(groups):
        interviewer_question = " ".join(group["interviewer"]).replace("Interviewer: ", "")
        embedding = model.encode(interviewer_question, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "interviewer_question": interviewer_question,
            "interviewee_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding,
            "speaking_round": index
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
                            api_key: str, logger: logging.Logger = None) -> list[dict]:
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
        summarized_question = summarize_question(original_question, gpt_model, api_key, logger)
        embedding = model.encode(summarized_question, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "original_question": original_question,
            "summarized_question": summarized_question,
            "original_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding
        })

    logger.info("Questions summarized and embedded.")
    logger.info("================================================================================================\n")
    return group_embeddings


def summarize_match_top_k_questions(guide_embedding: torch.Tensor, group_embeddings: list[dict], model: SentenceTransformer,
                                     device: torch.device, gpt_model: str, api_key: str, k: int = 3, logger: logging.Logger = None) -> list[dict]:
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

    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(guide_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["original_response"],
            "question": group["original_question"],
            "similarity": float(similarity)
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


def match_top_p_questions(guide_question: str, group_embeddings: list[dict], model: SentenceTransformer, 
                          device: torch.device, p: float = 0.5, max_matches: int = 5) -> list[dict]:
    """
    Match a guideline question to the top k groups based on similarity.
    
    Args:
        guide_question (str): The guide question to match.
        group_embeddings (list[dict]): List of embedded groups.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        p (float): Threshold for similarity.
        max_matches (int): Maximum number of matches to return
    
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
            "speaking_round": group["speaking_round"],
            "similarity": float(similarity)
        })
    
    # TODO: Add logic for max_matches
    ret_similarity = [s for s in similarities if s["similarity"]>=p]
    ret_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    return ret_similarity