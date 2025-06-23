import re
from typing import Union

def is_answer_empty_or_confused(text: str) -> bool:
    """
    Determine if the answer is empty, ambiguous, or doctoring (non-informative).

    Parameters:
    - text (str): The answer text.

    Returns:
    - bool: True if the text is effectively non-informative or evasive.
    """
    if not isinstance(text, str) or not text.strip():
        return True

    text_lower = text.lower()
    
    # Remove brackets and other punctuation, then clean up spaces
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    # Keywords indicating confusion or lack of knowledge
    confusion_keywords = [
        "i don't know", "i do not know", "not sure", "no idea", "unclear", 
        "can't say", "cannot say", "i'm confused", "i am confused",
        "none", "nothing", "n/a", "ambiguous", "unsure", "uncertain",
        "no relevant response found"
    ]

    # Phrases indicating doctoring - vague or generic statements without content
    doctoring_patterns = [
        r"that's a great question", 
        r"it's hard to say", 
        r"it depends on many factors", 
        r"there are many perspectives", 
        r"this is a complex issue", 
        r"it varies from case to case", 
        r"many people believe", 
        r"some might argue", 
        r"we must consider all sides", 
        r"it's subjective", 
        r"one could interpret this in different ways"
    ]

    # Direct keyword match (check both original and cleaned text)
    if any(kw in text_lower for kw in confusion_keywords):
        return True
    
    if any(kw in text_clean for kw in confusion_keywords):
        return True

    # Regex pattern match for doctoring
    for pattern in doctoring_patterns:
        if re.search(pattern, text_lower):
            return True

    return False

def build_prompt(metric: str, row: dict, chunk_list: list[str] = None) -> str:
    """
    Build a GPT prompt string for evaluating a specific metric.

    Parameters:
    - metric (str): Metric type ('relevance', 'faithfulness', 'precision', 'recall', 'correctness').
    - row (dict): Dictionary or Series with at least 'question', 'answer_rag', and optionally 'retrieved_context' or 'answer_ref'.
    - chunk_list (list[str], optional): List of dialogue chunks (for precision and recall).

    Returns:
    - str: A formatted prompt string to be passed to GPT.

    Raises:
    - ValueError: If an unknown metric is passed.
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

        # Check if both RAG and reference answers are empty/confused
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