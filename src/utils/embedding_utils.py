import logging
import asyncio
from sentence_transformers import SentenceTransformer, util
from src.utils.api_utils import summarize_question_async
from src.utils.output_utils import output_divider
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
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "speaking_round": group["speaking_round"]
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
            "similarity": float(similarity),
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "speaking_round": group["speaking_round"]
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]

async def summarize_embed_groups_async(groups: list[dict], model: SentenceTransformer, device: torch.device, gpt_model: str,
                                logger: logging.Logger = None) -> list[dict]:
    """
    Embed summarized interviewer questions from the groups asynchronously.

    Args:
        groups (list[dict]): List of groups containing interviewer and interviewee turns.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        gpt_model (str): The GPT model for summarization.
        logger (logging.Logger, optional): Logger instance for logging execution information.

    Returns:
        list[dict]: List of dictionaries with embeddings, summarized questions, and original texts.
    """

    group_embeddings = []
    semaphore = asyncio.Semaphore(10)  # Limit to 5 concurrent API calls

    async def summarize_with_limit(question):
        async with semaphore:
            return await summarize_question_async(question, gpt_model, logger)

    # Extract original questions
    original_questions = [" ".join(group["interviewer"]).replace("Interviewer: ", "") for group in groups]
    # Parallelize summarization of transcript questions
    tasks = [summarize_with_limit(q) for q in original_questions]
    summarized_questions = await asyncio.gather(*tasks, return_exceptions=True)

    for group, original_question, summarized_question in zip(groups, original_questions, summarized_questions):
        # Handle exceptions from asyncio.gather
        if isinstance(summarized_question, Exception):
            logger.error(f"Error summarizing question '{original_question}': {str(summarized_question)}")
            summarized_question = original_question  # Fallback to original question

        embedding = model.encode(summarized_question, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "original_question": original_question,
            "summarized_question": summarized_question,
            "original_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding,
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "speaking_round": group["speaking_round"]
        })

    logger.info("Questions summarized and embedded.")
    output_divider(logger, True)
    return group_embeddings

async def summarize_match_top_k_questions_async(guide_embedding: torch.Tensor, group_embeddings: list[dict], k: int = 3, logger: logging.Logger = None) -> list[dict]:
    """
    Match a summarized guideline question to the top k groups based on similarity asynchronously.

    Args:
        guide_embedding (torch.Tensor): The embedding of the guide question.
        group_embeddings (list[dict]): List of embedded groups.
        model (SentenceTransformer): The embedding model.
        device (torch.device): The device to use for computation.
        gpt_model (str): The GPT model for summarization.
        api_key (str): The OpenAI API key for summarization.
        k (int): Number of top matches to return.
        logger (logging.Logger, optional): Logger instance for logging execution information.

    Returns:
        list[dict]: Top k matches with similarity scores, using original texts.
    """
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(guide_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["original_response"],
            "question": group["original_question"],
            "similarity": float(similarity),
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "speaking_round": group["speaking_round"]
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


def match_top_p_questions(guide_embedding: torch.Tensor, group_embeddings: list[dict], 
                          p: float = 0.5, max_matches: int = 5) -> list[dict]:
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
    #question_embedding = model.encode(guide_question, convert_to_tensor=True, device=device)
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(guide_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["interviewee_response"],
            "question": group["interviewer_question"],
            "speaking_round": group["speaking_round"],
            "similarity": float(similarity),
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"]
        })
    
    ret_similarity = [s for s in similarities if s["similarity"]>=p]
    # do not sort and change order of speaking!
    #ret_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    return ret_similarity[:max_matches]