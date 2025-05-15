import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load OpenAI client and metric config
from src.utils.eval_config_utils import client, ACTIVE_METRICS

# GPT scoring helpers 

def ask_score_and_feedback(prompt: str, temperature: float = 0.0, model: str = "gpt-4.1-mini") -> tuple[float, str]:
    """
    Send a prompt to the GPT model to obtain a numerical score and feedback.

    Parameters:
    - prompt (str): The prompt string to evaluate.
    - temperature (float): Sampling temperature (default 0.0 for deterministic output).
    - model (str): Model name to use (default: "gpt-4o-mini").

    Returns:
    - tuple[float, str]: A tuple containing the numerical score and explanation string.
    """    
    messages = [
        {"role": "system", "content": "You are a helpful evaluation assistant. Respond in this format:\nScore: <number>\nFeedback: <short explanation>"},
        {"role": "user", "content": prompt}
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
    score = float(score_match.group(1).strip())
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return score, feedback


def build_prompt(metric: str, row: dict) -> str:
    """
    Construct a natural language prompt for GPT evaluation based on the scoring metric.

    Parameters:
    - metric (str): The metric name (e.g., 'relevance', 'faithfulness', etc.).
    - row (dict): A single row of input data containing question, answer, context, etc.

    Returns:
    - str: A formatted prompt string.
    """
    question = row["question"]
    answer = row["answer_rag"]
    context = row.get("retrieved_context", "")
    reference = row.get("answer_ref", "")

    if metric == "relevance":
        return f"""Evaluate the relevance of the answer to the question.
Question: {question}
Answer: {answer}
Rate from 1 (not relevant) to 5 (fully relevant). Explain briefly.
Score: X
Feedback: ..."""

    elif metric == "faithfulness":
        return f"""Evaluate the faithfulness of the answer to the retrieved context.
Context: {context}
Answer: {answer}
Rate from 1 (hallucinated) to 5 (fully grounded). Explain.
Score: X
Feedback: ..."""

    elif metric == "precision":
        return f"""Evaluate whether the context includes only necessary info to generate the answer.
Context: {context}
Answer: {answer}
Rate from 1 (verbose) to 5 (precise). Explain.
Score: X
Feedback: ..."""

    elif metric == "recall":
        return f"""Evaluate whether the context includes all necessary info to answer the question.
Question: {question}
Context: {context}
Answer: {answer}
Rate from 1 (missing info) to 5 (complete). Explain.
Score: X
Feedback: ..."""

    elif metric == "correctness":
        return f"""Compare the generated answer with the reference.
Question: {question}
Answer: {answer}
Reference: {reference}
Rate from 1 (wrong) to 5 (semantically equivalent). Explain.
Score: X
Feedback: ..."""

    else:
        raise ValueError(f"Unknown metric: {metric}")


def score_ragas(row: pd.Series) -> pd.Series:
    """
    Evaluate a single row using the enabled metrics in ACTIVE_METRICS and GPT scoring.

    Parameters:
    - row (pd.Series): A row from the merged DataFrame containing question, context, etc.

    Returns:
    - pd.Series: A series of new columns with score and feedback for each metric.
    """
    metrics = []

    if pd.notna(row.get("retrieved_context")):
        for m in ["faithfulness", "precision", "recall"]:
            if ACTIVE_METRICS.get(m, False):
                metrics.append(m)

    if ACTIVE_METRICS.get("relevance", False):
        metrics.append("relevance")

    if ACTIVE_METRICS.get("correctness", False) and pd.notna(row.get("answer_ref")):
        metrics.append("correctness")

    results = {}
    for metric in metrics:
        prompt = build_prompt(metric, row)
        score, feedback = ask_score_and_feedback(prompt)
        results[f"{metric}_score"] = score
        results[f"{metric}_feedback"] = feedback

    return pd.Series(results)


# Main evaluation function 

def run_ragas_evaluation(
    rag_path: str,
    ref_path: str,
    context_path: str,
    output_path: str
):
    """
    Execute the full RAGAS evaluation pipeline:
    1. Load RAG answers, reference answers, and retrieved contexts.
    2. Merge them into a single DataFrame.
    3. Score each row using GPT across selected metrics.
    4. Save the result with scores and feedback to CSV.

    Parameters:
    - rag_path (str): Path to the RAG-generated answers CSV.
    - ref_path (str): Path to the human reference answers CSV.
    - context_path (str): Path to the cleaned retrieved contexts CSV.
    - output_path (str): Path to save the final output CSV with scores.
    """
    rag_df = pd.read_csv(os.path.expanduser(rag_path))
    ref_df = pd.read_csv(os.path.expanduser(ref_path))
    context_df = pd.read_csv(os.path.expanduser(context_path))

    # Clean column names
    rag_df.columns = rag_df.columns.str.strip()
    ref_df.columns = ref_df.columns.str.strip()
    context_df.columns = context_df.columns.str.strip()

    rag_df = rag_df.rename(columns={"Interview File": "respondent_id"})
    ref_df = ref_df.rename(columns={"respondent_id": "respondent_id"})
    context_df = context_df.rename(columns={"guide_question": "question"})

    # Melt and merge
    rag_long = rag_df.melt(id_vars=["respondent_id"], var_name="question", value_name="answer_rag")
    ref_long = ref_df.melt(id_vars=["respondent_id"], var_name="question", value_name="answer_ref")
    merged_df = pd.merge(rag_long, ref_long, on=["respondent_id", "question"], how="left")
    merged_df = pd.merge(merged_df, context_df, on=["respondent_id", "question"], how="left")

    test_df = merged_df.head(10)

    # Evaluate
    tqdm.pandas()
    scores_df = test_df.progress_apply(score_ragas, axis=1)
    result_df = pd.concat([test_df, scores_df], axis=1)
    result_df.to_csv(os.path.expanduser(output_path), index=False)
    print(f"\n Evaluation completed. Saved to: {output_path}")


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation pipeline.")
    parser.add_argument("--rag", required=True, help="Path to RAG-generated answer CSV file.")
    parser.add_argument("--ref", required=True, help="Path to reference answer CSV file.")
    parser.add_argument("--context", required=True, help="Path to retrieved context CSV file.")
    parser.add_argument("--out", required=True, help="Output path for evaluation results.")

    args = parser.parse_args()

    run_ragas_evaluation(
        rag_path=args.rag,
        ref_path=args.ref,
        context_path=args.context,
        output_path=args.out
    )
