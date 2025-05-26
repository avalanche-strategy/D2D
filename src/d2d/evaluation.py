import pandas as pd
from src.utils.log_utils import extract_retrieved_contexts
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.utils.eval_config_utils import client, ACTIVE_METRICS


class D2DEvaluator:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.0):
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

        return results
