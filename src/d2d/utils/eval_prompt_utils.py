from typing import Union
from sentence_transformers import SentenceTransformer, util
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SIMILARITY_THRESHOLD = 0.78  # Default threshold for detecting confused/empty answers

# Load embedding model globally
_model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
_model = SentenceTransformer(_model_name)
_confused_templates = [
    "I don't know", "Not sure", "No idea", "Cannot say",
    "I'm confused", "This is unclear", "There is no information",
    "N/A", "None", "No relevant response found"
]
_confused_embeddings = _model.encode(_confused_templates, convert_to_tensor=True, normalize_embeddings=True)


def is_answer_empty_or_confused(text: Union[str, None], threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """
    Determine if the answer is empty, ambiguous, or evasive using embedding similarity.

    Parameters:
    - text (str): The answer text.
    - threshold (float): Cosine similarity threshold. Default = 0.78 for multi-qa-mpnet.

    Returns:
    - bool: True if the text is semantically similar to a confused/non-informative template.
    """
    if not isinstance(text, str) or not text.strip():
        return True

    # Normalize and embed input
    embedding = _model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    cosine_score = util.cos_sim(embedding, _confused_embeddings).max().item()
    return cosine_score > threshold


def build_prompt(metric: str, row: dict, chunk_list: list[str] = None) -> str:
    """
    Build a GPT prompt string for evaluating a specific metric.

    Parameters:
    - metric (str): Metric type ('relevance', 'faithfulness', 'precision', 'recall', 'correctness').
    - row (dict): Dictionary or Series with 'question', 'answer_rag', 'answer_ref', etc.
    - chunk_list (list[str], optional): List of dialogue chunks (for some metrics).

    Returns:
    - str: A formatted prompt string to be passed to LLM.
    """
    question = row["question"]
    answer = row["answer_rag"]
    answer_ref = row["answer_ref"]
    context = row.get("retrieved_context", "")
    reference = row.get("answer_ref", "")

    if metric == "relevance":
        return f"""Evaluate the relevance of the answer to the question.
Question: {question}
Answer: {answer}
Rate from 1 (not relevant) to 5 (fully relevant). Explain briefly.
Score: <value between 1 and 5>
Feedback: ..."""

    elif metric == "faithfulness":
        chunks = "\n\n".join(chunk_list or [])
        rag_confused = is_answer_empty_or_confused(answer)
        ref_confused = is_answer_empty_or_confused(answer_ref)

        if not rag_confused:
            return f"""You are evaluating the **faithfulness** of a generated answer with respect to the retrieved context.

Context: 
{chunks}

Answer: 
{answer}

Instructions:
1. Extract all factual claims made in the answer.
2. For each fact, determine whether it is supported by the context.
3. If a fact is not present in the context, it is a hallucination.
4. Compute:  
   faithfulness = (number of supported facts) / (total number of facts stated in the answer)

Then clearly report:

Score: <value between 0 and 1>  
Feedback:  
- Total facts: ...
- Supported facts: ...  
- Hallucinated facts: ...  
- Explanation: ...

Score:  
Feedback:
"""

        elif rag_confused and ref_confused:
            return f"""You are evaluating the **faithfulness** of a generated answer with respect to the retrieved context.

The answer is ambiguous or refuses to answer (e.g., "No relevant response found"), and the reference answer is also ambiguous.

In such cases, judge whether the **tone and attitude** of the answer are consistent with the context.
- Score 1.0 → if the answer is cautious and aligned with the context
- Score 0.5 → if it's neutral or unclear
- Score 0.0 → if the answer misrepresents the attitude or implies something untrue

Context:
{chunks}

Answer:
{answer}

Then clearly report:
Score: <value between 0 and 1>
Feedback: Explain your reasoning.

Score:
Feedback:
"""

        elif rag_confused and not ref_confused:
            return f"""You are evaluating the **faithfulness** of a generated answer with respect to the retrieved context.

The generated answer is ambiguous or refuses to answer (e.g., "No relevant response found"), but the reference answer contains concrete information.

In this case, the answer failed to reflect key information from the context and should be penalized.

Give a score of 0.0 and explain briefly.

Context:
{chunks}

Answer:
{answer}

Reference:
{answer_ref}

Score:
Feedback: The model failed to reflect key information available in the reference.
"""

    elif metric == "precision":
        chunks = "\n\n".join(chunk_list or [])
        rag_confused = is_answer_empty_or_confused(answer)
        ref_confused = is_answer_empty_or_confused(answer_ref)

        if rag_confused and ref_confused:
            return f"""You are evaluating the **precision** of retrieved chunks with respect to the reference answer.

Each chunk is a dialogue pair from the original interview, labeled as "chunk 1", "chunk 2", etc.

Chunks:
{chunks}

Reference Answer:
{answer_ref}

Instructions:
- Identify which chunks are actually relevant to the reference answer.
- Report only the chunk numbers that are relevant or irrelevant.
- Calculate precision = (number of relevant chunks) / (total retrieved chunks).
- Then clearly report the score and feedback in this format:

Score: <value between 0 and 1>
Feedback:
Used chunks: chunk N, chunk M, ...
Unused chunks: chunk X, chunk Y, ...
Explanation: <brief justification>

Score: 
Feedback: """
        else:
            return f"""You are evaluating the **precision** of retrieved chunks in relation to the generated answer.

Each chunk is a dialogue pair from the original interview, labeled as "chunk 1", "chunk 2", etc.

Chunks:
{chunks}

Answer:
{answer}

Instructions:
- Identify which chunks were actually used to generate this answer.
- Report only the chunk numbers that are used or unused.
- Calculate precision = (number of used chunks) / (total retrieved chunks).
- Then clearly report the score and feedback in this format:

Score: <value between 0 and 1>
Feedback:
Used chunks: chunk N, chunk M, ...
Unused chunks: chunk X, chunk Y, ...
Explanation: <brief justification>

Score: 
Feedback: """

    elif metric == "recall":
        chunks = "\n\n".join(chunk_list or [])
        rag_confused = is_answer_empty_or_confused(answer)
        ref_confused = is_answer_empty_or_confused(answer_ref)

        if rag_confused and ref_confused:
            return f"""You are evaluating the **recall** of retrieved chunks with respect to the reference answer.

Question:
{question}

Reference Answer:
{answer_ref}

Chunks:
{chunks}

Instructions:
- Identify the key facts needed to answer this question based on the reference answer.
- Determine which of these facts appear in the retrieved chunks.
- Calculate recall = (covered facts) / (total facts required).
- Provide:
  - Score (between 0 and 1),
  - Feedback listing: 'Covered facts: ...' and 'Uncovered facts: ...'

Score: 
Feedback: """
        else:
            return f"""You are evaluating the **recall** of retrieved chunks with respect to the generated answer.

Question:
{question}

Answer:
{answer}

Chunks:
{chunks}

Instructions:
- Identify the key facts needed to answer this question.
- Determine which of these facts appear in the retrieved chunks.
- Calculate recall = (covered facts) / (total facts required).
- Provide:
  - Score (between 0 and 1),
  - Feedback listing: 'Covered facts: ...' and 'Uncovered facts: ...'

Score: 
Feedback: """

    elif metric == "correctness":
        if is_answer_empty_or_confused(answer) and is_answer_empty_or_confused(reference):
            return f"""Both the generated answer and the reference answer indicate that no relevant response was possible.
            
Question: {question}
Answer: {answer}
Reference: {reference}

In this case, treat the generated answer as semantically equivalent to the reference.

Score: 5.0  
Feedback: The generated answer correctly matches the reference in indicating no relevant information was found.
"""
        return f"""Compare the generated answer with the reference.
Question: {question}
Answer: {answer}
Reference: {reference}
Rate from 1 (wrong) to 5 (semantically equivalent). Explain.
Score: <value between 1 and 5>
Feedback: ..."""

    else:
        raise ValueError(f"Unknown metric: {metric}")
