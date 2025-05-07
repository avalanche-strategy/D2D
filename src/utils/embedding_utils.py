from sentence_transformers import SentenceTransformer, util
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