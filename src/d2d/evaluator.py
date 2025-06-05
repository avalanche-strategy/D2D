import pandas as pd
import os
from .utils.log_utils import extract_retrieved_contexts
from .evaluation.ragas_eval import run_ragas_evaluation
from .utils.eval_config_utils import client, ACTIVE_METRICS


class D2DEvaluator:
    """
    D2DEvaluator is a pipeline orchestrator for evaluating outputs from a 
    Retrieval-Augmented Generation (RAG) system against reference answers 
    using GPT-based scoring.

    It provides two main functions:
    1. `evaluate`: Extracts context, runs RAGAS evaluation, and saves results.
    2. `post_process_results`: Highlights low scores and computes a joint 
       performance metric across multiple evaluation dimensions.

    Attributes:
        METRICS (List[str]): List of metric names used in evaluation scoring.
        model (str): Name of the LLM model used for evaluation scoring (default "gpt-4o-mini").
        temperature (float): Temperature setting for GPT-based evaluation prompts.
    """
    METRICS = ["faithfulness", "correctness", "precision", "recall", "relevance"]

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_concurrent_calls=5):
        """
        Initialize the evaluator with model and decoding temperature.

        Args:
            model (str): LLM model name to use for evaluation.
            temperature (float): Decoding temperature for GPT prompts.
            max_concurrent_calls (int): Maximum number of concurrent API calls to the LLM.
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent_calls = max_concurrent_calls

    def evaluate(
        self,
        log_input_path: str,
        context_output_path: str,
        rag_csv_path: str,
        ref_csv_path: str,
        eval_output_path: str
    ) -> pd.DataFrame:
        """
        Orchestrates the evaluation pipeline:
        1. Extracts retrieved context from log files.
        2. Runs RAGAS evaluation on the model answers against reference answers.
        3. Saves the evaluation results to CSV.

        Args:
            log_input_path (str): Path to the generator log file.
            context_output_path (str): Path where the extracted context CSV will be saved.
            rag_csv_path (str): Path to the model-generated answers CSV.
            ref_csv_path (str): Path to the human-annotated reference answers CSV.
            eval_output_path (str): Path where the evaluation result CSV will be saved.

        Returns:
            pd.DataFrame: A DataFrame containing the full evaluation results with metric scores.
        """
        # Make sure output directory exists
        os.makedirs(os.path.dirname(context_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

        # Step 1: Extract retrieved context
        extract_retrieved_contexts(log_input_path, context_output_path)

        # Step 2: Run evaluation
        results = run_ragas_evaluation(
            rag_path=rag_csv_path,
            ref_path=ref_csv_path,
            context_path=context_output_path,
            output_path=eval_output_path,
            max_concurrent_calls=self.max_concurrent_calls
        )

        results = pd.read_csv(eval_output_path)
        return results

    def post_process_results(
        self,
        results: pd.DataFrame,
        weights: dict,
        output_prefix: str = "post_eval",
        highlight_threshold: float = 1.0
    ) -> None:
        """
        Post-processes evaluation results by:
        1. Highlighting rows with any score below or equal to a threshold.
        2. Computing a joint metric score using a weighted average of individual metrics.

        Saves two CSVs:
        - `<output_prefix>_highlighted.csv`: Rows with low scores labeled.
        - `<output_prefix>_joint_metric.csv`: Respondent-level joint scores.

        Args:
            results (pd.DataFrame): DataFrame containing evaluation results.
            weights (dict): Dictionary assigning weights to each metric; must sum to 1.0.
            output_prefix (str): Prefix for output file names (default: "post_eval").
            highlight_threshold (float): Threshold under which scores are considered low (default: 1.0).

        Raises:
            ValueError: If weights are missing or do not sum to 1.0.
        """
        highlight_rows = []
        for idx, row in results.iterrows():
            row_copy = row.copy()
            found = False
            for metric in self.METRICS:
                score_col = f"{metric}_score"
                score_val = row_copy.get(score_col)
                if pd.notna(score_val) and score_val <= highlight_threshold:
                    row_copy[score_col] = f"low-{score_val}"
                    found = True
            if found:
                highlight_rows.append(row_copy)

        highlight_df = pd.DataFrame(highlight_rows)
        highlight_path = f"{output_prefix}_highlighted.csv"
        highlight_df.to_csv(highlight_path, index=False)
        print(f"Highlighted rows saved to: {highlight_path}")

        for metric in self.METRICS:
            results[metric + "_score"] = pd.to_numeric(results[metric + "_score"], errors="coerce")

        if not weights:
            raise ValueError("Weights dictionary must be provided to compute joint score.")

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"The sum of weights must be exactly 1.0. Currently: {total_weight}")

        group = results.groupby("respondent_id")[ [f"{m}_score" for m in self.METRICS] ].mean().reset_index()
        group["joint_score"] = sum(group[f"{m}_score"] * weights.get(m, 0.0) for m in self.METRICS)
        group = group.sort_values(by="joint_score", ascending=False)

        joint_path = f"{output_prefix}_joint_metric.csv"
        group.to_csv(joint_path, index=False, float_format="%.1f")
        print(f"Joint metric summary saved to: {joint_path}")
