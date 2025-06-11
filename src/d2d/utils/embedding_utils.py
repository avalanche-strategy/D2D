import logging
import asyncio
from sentence_transformers import SentenceTransformer, util
from .api_utils import summarize_question_async

import torch

async def summarize_embed_groups_async(groups: list[dict], model: SentenceTransformer, device: torch.device, gpt_model: str,
                                logger: logging.Logger = None) -> list[dict]:
    """
    Summarize and embed interviewer questions and interviewee responses from groups asynchronously.

    Args:
        groups (list[dict]): List of dictionaries containing interviewer and interviewee turns.
        model (SentenceTransformer): The embedding model for generating embeddings.
        device (torch.device): The PyTorch device for computation (e.g., CPU or GPU).
        gpt_model (str): The LLM model name for question summarization (compatible with LiteLLM).
        logger (logging.Logger, optional): Logger instance for logging execution information.

    Returns:
        list[dict]: List of dictionaries, each containing:
            - original_question (str): The original interviewer question.
            - summarized_question (str): The summarized interviewer question.
            - original_response (str): The original interviewee response.
            - embedding (torch.Tensor): Embedding of the summarized question.
            - response_embedding (torch.Tensor): Embedding of the interviewee response.
            - interviewer_line_ref (int): Line number of the interviewer’s turn.
            - interviewee_line_ref (int): Line number of the interviewee’s turn.
            - speaking_round (int): The conversation round index.
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

    # check for task errors
    error_results = [e for e in summarized_questions if isinstance(e, Exception)]
    good_results = [r for r in summarized_questions if not isinstance(r, Exception)]

    if len(good_results)==0 and len(error_results)>0:
        error_list = set([e.__class__.__qualname__ for e in error_results])
        logger.error(f"All Question Summarizing tasks returned errors of type: {error_list}")
        raise error_results[0]
    elif len(error_results)>0:
        logger.warning("Some Question Summarizing tasks in have errors.")

    for group, original_question, summarized_question in zip(groups, original_questions, summarized_questions):
        # Handle exceptions from asyncio.gather
        if isinstance(summarized_question, Exception):
            logger.error(f"Error summarizing question '{original_question}': {str(summarized_question)}")
            logger.warning(f"Switching to original question '{original_question}'")
            summarized_question = original_question  # Fallback to original question

        embedding = model.encode(summarized_question, convert_to_tensor=True, device=device)
        # embed the response to enable semantic match in reference line search
        interviewee_response = " ".join(group["interviewee"]).replace("Interviewee: ", "")
        response_embedding = model.encode(interviewee_response, convert_to_tensor=True, device=device)
        group_embeddings.append({
            "original_question": original_question,
            "summarized_question": summarized_question,
            "original_response": " ".join(group["interviewee"]).replace("Interviewee: ", ""),
            "embedding": embedding,
            "response_embedding": response_embedding,
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "speaking_round": group["speaking_round"]
        })

    logger.info("Questions summarized and embedded.")

    # Importing here to avoid circular import error
    from .output_utils import output_divider
    output_divider(logger, True)

    return group_embeddings

async def summarize_match_top_k_questions_async(guide_embedding: torch.Tensor, group_embeddings: list[dict], k: int = 3) -> list[dict]:
    """
    Match a summarized guideline question to the top k groups based on cosine similarity.

    Args:
        guide_embedding (torch.Tensor): The embedding of the guideline question.
        group_embeddings (list[dict]): List of dictionaries with embedded group data.
        k (int): Number of top matches to return (default: 3).

    Returns:
        list[dict]: List of up to k dictionaries, each containing:
            - response (str): The original interviewee response.
            - question (str): The original interviewer question.
            - similarity (float): Cosine similarity score.
            - interviewer_line_ref (int): Line number of the interviewer’s turn.
            - interviewee_line_ref (int): Line number of the interviewee’s turn.
            - speaking_round (int): The conversation round index.
            - response_embedding (torch.Tensor): Embedding of the interviewee response.
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
            "speaking_round": group["speaking_round"],
            "response_embedding": group["response_embedding"],

        })
        
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


async def summarize_match_top_p_questions_async(guide_embedding: torch.Tensor, group_embeddings: list[dict],
                          p: float = 0.5, max_matches: int = 5) -> list[dict]:
    """
    Match a guideline question to groups with similarity above a threshold.

    Args:
        guide_embedding (torch.Tensor): The embedding of the guideline question.
        group_embeddings (list[dict]): List of dictionaries with embedded group data.
        p (float): Minimum cosine similarity threshold (default: 0.5).
        max_matches (int): Maximum number of matches to return (default: 5).

    Returns:
        list[dict]: List of dictionaries for matches with similarity >= p, up to max_matches, each containing:
            - response (str): The original interviewee response.
            - question (str): The original interviewer question.
            - similarity (float): Cosine similarity score.
            - interviewer_line_ref (int): Line number of the interviewer’s turn.
            - interviewee_line_ref (int): Line number of the interviewee’s turn.
            - speaking_round (int): The conversation round index.
            - response_embedding (torch.Tensor): Embedding of the interviewee response.
    """
    #question_embedding = model.encode(guide_question, convert_to_tensor=True, device=device)
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(guide_embedding, group["embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "response": group["original_response"],
            "question": group["original_question"],
            "speaking_round": group["speaking_round"],
            "similarity": float(similarity),
            "interviewer_line_ref": group["interviewer_line_ref"],
            "interviewee_line_ref": group["interviewee_line_ref"],
            "response_embedding": group["response_embedding"],
        })
    
    ret_similarity = [s for s in similarities if s["similarity"]>=p]
    # do not sort and change the order of speaking!
    #ret_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    return ret_similarity[:max_matches]


def match_top_responses(model: SentenceTransformer,
                        device: torch.device,
                        logger: logging.Logger,
                        extracted_phrase: str, group_embeddings: list[dict],
                        p_threshold: float = 0.8, max_matches: int = 3) -> list[dict]:
    """
    Match an extracted phrase to interviewee responses based on similarity for reference line search.

    Args:
        model (SentenceTransformer): The embedding model for generating embeddings.
        device (torch.device): The PyTorch device for computation (e.g., CPU or GPU).
        logger (logging.Logger): Logger instance for logging execution information.
        extracted_phrase (str): The phrase to match against interviewee responses.
        group_embeddings (list[dict]): List of dictionaries with embedded group data.
        p_threshold (float): Minimum cosine similarity threshold (default: 0.8).
        max_matches (int): Maximum number of matches to return (default: 3).

    Returns:
        list[dict]: List of dictionaries for matches with similarity >= p_threshold, up to max_matches, each containing:
            - similarity (float): Cosine similarity score.
            - interviewee_line_ref (int): Line number of the interviewee’s turn.
            - response (str): The original interviewee response.
    """
    query_embedding = model.encode(extracted_phrase, convert_to_tensor=True, device=device)
    similarities = []
    for group in group_embeddings:
        similarity = util.cos_sim(query_embedding, group["response_embedding"]).cpu().numpy()[0][0]
        similarities.append({
            "similarity": float(similarity),
            "interviewee_line_ref": group["interviewee_line_ref"],
            "response": group["response"]
        })
    logger.info(f"Similarities for '{extracted_phrase}'")
    logger.info([round(s["similarity"], 3) for s in similarities])
    ret_similarity = [s for s in similarities if s["similarity"] >= p_threshold]
    # do not sort and change order of speaking!
    # ret_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    return ret_similarity[:max_matches]
