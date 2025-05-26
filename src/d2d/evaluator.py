import pandas as pd
from src.utils.log_utils import extract_retrieved_contexts
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.utils.eval_config_utils import client, ACTIVE_METRICS


class D2DEvaluator:

    METRICS = ["faithfulness", "correctness", "precision", "recall", "relevance"]

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        log_input_path: str,
        context_output_path: str,
        rag_csv_path: str,
        ref_csv_path: str,
        eval_output_path: str
    ) -> pd.DataFrame:
        """
        End-to-end evaluation pipeline:
        1. Extract retrieved context from log
        2. Run RAGAS evaluation using GPT
        3. Save output and return results

        Returns:
            pd.DataFrame: Evaluation results
        """
        # Step 1: Extract retrieved context
        extract_retrieved_contexts(log_input_path, context_output_path)

        # Step 2: Run evaluation
        results = run_ragas_evaluation(
            rag_path=rag_csv_path,
            ref_path=ref_csv_path,
            context_path=context_output_path,
            output_path=eval_output_path
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
