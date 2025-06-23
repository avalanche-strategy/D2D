import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load OpenAI client and metric config
from d2d.utils.eval_config_utils import client, ACTIVE_METRICS
from concurrent.futures import ThreadPoolExecutor, as_completed


# GPT scoring helpers

def ask_score_and_feedback(prompt: str, temperature: float = 0.0, model: str = "gpt-4.1-mini", is_proportion: bool = False) -> tuple[float, str]:
    """
    Send a prompt to a GPT model and extract the score and feedback.

    Parameters:
    - prompt (str): The prompt string to be evaluated by GPT.
    - temperature (float): Decoding temperature. Default is 0.0 for deterministic output.
    - model (str): Model name to use for the request.
    - is_proportion (bool): Whether the score is expected to be a float in [0, 1], 
                            which will be mapped to a 1–5 scale.

    Returns:
    - tuple[float, str]: Scaled score (1–5 if is_proportion=True) and feedback string.

    Raises:
    - ValueError: If no score can be parsed from GPT response.
    """
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Invalid prompt: expected non-empty string, got {type(prompt).__name__}: {prompt}")
    
    messages = [
        {"role": "system", "content": "You are a helpful evaluation assistant. Respond in this format:\nScore: <number between 0 and 1>\nFeedback: <explanation>"},
        {"role": "user", "content": prompt.strip()}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    content = response.choices[0].message.content.strip()
    content = content.replace("\\n", "\n").replace("\r", "\n")

    score_match = re.search(r"Score\s*[:：]\s*([0-9.]+)", content, re.IGNORECASE)
    feedback_match = re.search(r"Feedback\s*[:：]\s*(.*)", content, re.IGNORECASE | re.DOTALL)

    if not score_match:
        raise ValueError(f"Could not find Score in response:\n{content}")
    
    raw_score = float(score_match.group(1).strip())
    score = round(1 + 4 * raw_score, 1) if is_proportion else raw_score
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return score, feedback


def split_chunks(context: str) -> list[str]:
    """
    Split a context string into individual chunks using 'chunk N:' as delimiter.

    Parameters:
    - context (str): The full context text as a string with 'chunk N:' labels.

    Returns:
    - list[str]: List of chunk strings, each including its own 'chunk N:' header and content.
    """
    if not isinstance(context, str) or not context.strip():
        return []

    # Match from 'chunk N:' to before the next 'chunk M:' or end of text
    pattern = r"(chunk \d+:.*?)(?=chunk \d+:|$)"
    matches = re.findall(pattern, context.strip(), flags=re.DOTALL | re.IGNORECASE)
    return [chunk.strip() for chunk in matches if chunk.strip()]



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


def score_ragas(row: pd.Series) -> pd.Series:
    """
    Apply GPT scoring for each enabled metric on a single row of input data.

    Metrics include relevance, faithfulness, correctness, and optionally precision/recall using chunk-level evaluation.

    Parameters:
    - row (pd.Series): A merged input row containing question, answer, context, and reference.

    Returns:
    - pd.Series: Contains new columns like '<metric>_score' and '<metric>_feedback'.
    """
    results = {}
    metrics = []

    # Clean & check fields
    answer_ref = row.get("answer_ref")
    retrieved_context = row.get("retrieved_context")

    has_context = pd.notna(retrieved_context) and isinstance(retrieved_context, str) and retrieved_context.strip()
    has_ref = True

    # Always allow correctness and relevance if enabled and valid
    if ACTIVE_METRICS.get("relevance", False):
        metrics.append("relevance")
    if ACTIVE_METRICS.get("correctness", False):
        metrics.append("correctness")

    # Add context-related metrics only if retrieved context exists
    if has_context:
        for m in ["faithfulness", "precision", "recall"]:
            if ACTIVE_METRICS.get(m, False):
                metrics.append(m)

    chunk_list = split_chunks(retrieved_context) if has_context else []

    if not has_context:
        # Provide fallback values for context-dependent metrics
        if ACTIVE_METRICS.get("faithfulness", False):
            results["faithfulness_score"] = 1.0
            results["faithfulness_feedback"] = "No retrieved context. Cannot verify any factual claims. All considered hallucinated."

        if ACTIVE_METRICS.get("recall", False):
            results["recall_score"] = 1.0
            results["recall_feedback"] = "No context available. None of the required information is covered."

        if ACTIVE_METRICS.get("precision", False):
            results["precision_score"] = np.nan
            results["precision_feedback"] = "No retrieved chunks. Precision is not applicable."

    # Main evaluation loop
    for metric in metrics:
        try:
            # Handle special case for correctness when both answers are empty/confused
            if metric == "correctness":
                rag_answer = row.get("answer_rag", "")
                ref_answer = row.get("answer_ref", "")
                
                if is_answer_empty_or_confused(rag_answer) and is_answer_empty_or_confused(ref_answer):
                    results["correctness_score"] = 5.0
                    results["correctness_feedback"] = "The generated answer correctly matches the reference in indicating no relevant information was found."
                    continue

            use_chunks = metric in ["precision", "recall", "faithfulness"]
            prompt = build_prompt(metric, row, chunk_list=chunk_list if use_chunks else None)
        
            if not prompt or not prompt.strip():
                print(f"Warning: Empty prompt for metric {metric} in row {row.name}")
                results[f"{metric}_score"] = None
                results[f"{metric}_feedback"] = f"Failed to generate prompt for {metric}"
                continue
            
            is_proportion_metric = metric in ["faithfulness", "precision", "recall"] 
            score, feedback = ask_score_and_feedback(prompt, is_proportion=is_proportion_metric)
            results[f"{metric}_score"] = score
            results[f"{metric}_feedback"] = feedback
        
        except Exception as e:
            print(f"Error evaluating {metric} for row {row.name}: {e}")
            results[f"{metric}_score"] = None
            results[f"{metric}_feedback"] = f"Error: {str(e)}"


    return pd.Series(results)



def run_ragas_evaluation(rag_path: str, ref_path: str, context_path: str, output_path: str, max_concurrent_calls: int = 5):
    """
    Evaluate RAG-generated answers using GPT-based metrics and save results.

    Loads RAG, reference, and context CSVs, applies relevance, faithfulness,
    precision, recall, and correctness metrics in parallel, and writes output to CSV.

    Parameters:
    - rag_path (str): Path to RAG answers CSV.
    - ref_path (str): Path to reference answers CSV.
    - context_path (str): Path to context CSV.
    - output_path (str): Output CSV path.
    - max_concurrent_calls (int): Max concurrent GPT calls.

    Returns:
    - None
    """
    rag_df = pd.read_csv(os.path.expanduser(rag_path))
    ref_df = pd.read_csv(os.path.expanduser(ref_path))
    context_df = pd.read_csv(os.path.expanduser(context_path))

    rag_df.columns = rag_df.columns.str.strip()
    ref_df.columns = ref_df.columns.str.strip()
    context_df.columns = context_df.columns.str.strip()

    rag_df = rag_df.rename(columns={"Interview File": "respondent_id"})
    ref_df = ref_df.rename(columns={"respondent_id": "respondent_id"})
    context_df = context_df.rename(columns={"guide_question": "question"})

    rag_long = rag_df.melt(id_vars=["respondent_id"], var_name="question", value_name="answer_rag")
    ref_long = ref_df.melt(id_vars=["respondent_id"], var_name="question", value_name="answer_ref")
    merged_df = pd.merge(rag_long, ref_long, on=["respondent_id", "question"], how="left")
    merged_df = pd.merge(merged_df, context_df, on=["respondent_id", "question"], how="left")

    print(f"Scoring {len(merged_df)} examples with up to {max_concurrent_calls} concurrent threads...")

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
        futures = {executor.submit(score_ragas, row): idx for idx, row in merged_df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Error in row {idx}: {e}")
                results.append((idx, pd.Series())) 

    scores_df = pd.DataFrame([r[1] for r in sorted(results, key=lambda x: x[0])])
    result_df = pd.concat([merged_df.reset_index(drop=True), scores_df], axis=1)

    result_df.to_csv(os.path.expanduser(output_path), index=False)
    print(f"\nEvaluation completed. Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation pipeline.")
    parser.add_argument("--max_concurrent_calls", type=int, default=5, help="Max number of concurrent LLM API calls.")
    parser.add_argument("--rag", required=True, help="Path to RAG-generated answer CSV file.")
    parser.add_argument("--ref", required=True, help="Path to reference answer CSV file.")
    parser.add_argument("--context", required=True, help="Path to retrieved context CSV file.")
    parser.add_argument("--out", required=True, help="Output path for evaluation results.")

    args = parser.parse_args()

    run_ragas_evaluation(
        max_concurrent_calls=args.max_concurrent_calls,
        rag_path=args.rag,
        ref_path=args.ref,
        context_path=args.context,
        output_path=args.out
    )
