import sys
import os
import time

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.d2d import D2DEvaluator

def main_eval(
    log_input_path: str,
    context_output_path: str,
    rag_csv_path: str,
    ref_csv_path: str,
    eval_output_path: str,
    post_eval_prefix: str,
    weights: dict = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_concurrent_calls: int = 5
):
    if weights is None:
        weights = {
            "correctness": 0.3,
            "faithfulness": 0.2,
            "precision": 0.2,
            "recall": 0.2,
            "relevance": 0.1
        }

    evaluator = D2DEvaluator(
        model=model,
        temperature=temperature,
        max_concurrent_calls=max_concurrent_calls
    )

    results = evaluator.evaluate(
        log_input_path=log_input_path,
        context_output_path=context_output_path,
        rag_csv_path=rag_csv_path,
        ref_csv_path=ref_csv_path,
        eval_output_path=eval_output_path
    )

    evaluator.post_process_results(
        results=results,
        weights=weights,
        output_prefix=post_eval_prefix
    )

# Example 1: Default test setup
def main_default_eval():
    model = "gpt-4o-mini"
    temperature = 0.0
    max_concurrent_calls = 5
    weights = None

    log_input = "results/D2D_survey_food_generator_log_2025-05-30_09-45.txt"
    context_output = "eval_results/retrieved_contexts.csv"
    rag_csv = "results/D2D_survey_food_responses_2025-05-30_09-45.csv"
    ref_csv = "data/private_data/references/responses_food.csv"
    eval_output = "eval_results/eval_output.csv"
    post_eval_prefix = "eval_results/eval_output_post"

    main_eval(
        model=model,
        temperature=temperature,
        max_concurrent_calls=max_concurrent_calls,
        log_input_path=os.path.expanduser(log_input),
        context_output_path=os.path.expanduser(context_output),
        rag_csv_path=os.path.expanduser(rag_csv),
        ref_csv_path=os.path.expanduser(ref_csv),
        eval_output_path=os.path.expanduser(eval_output),
        post_eval_prefix=os.path.expanduser(post_eval_prefix),
        weights=weights
    )

if __name__ == "__main__":
    start = time.time()
    main_default_eval()
    end = time.time()
    print(f"Evaluation completed in {end - start:.2f} seconds.")