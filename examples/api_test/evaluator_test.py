import sys
import os
import time

# Add the root directory (D2D/) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.d2d import D2DEvaluator

def main_eval():
    evaluator = D2DEvaluator(
        model="gpt-4o-mini",
        temperature=0.0
    )

    # Define paths
    log_input_path = os.path.join(root_dir, "results", "D2D_survey_48_generator_log_2025-05-26_14-00.txt")
    context_output_path = os.path.join(root_dir, "eval_results", "retrieved_contexts.csv")
    rag_csv_path = os.path.join(root_dir, "results", "D2D_survey_48_responses_2025-05-26_14-00.csv")
    ref_csv_path = os.path.join(root_dir, "data", "private_data", "references", "responses_48.csv")
    eval_output_path = os.path.join(root_dir, "eval_results", "eval_output.csv")

    # Start processing evaluation
    results = evaluator.evaluate(
        log_input_path=log_input_path,
        context_output_path=context_output_path,
        rag_csv_path=rag_csv_path,
        ref_csv_path=ref_csv_path,
        eval_output_path=eval_output_path
    )

    # Post-process results
    weights = {
        "correctness": 0.3,
        "faithfulness": 0.2,
        "precision": 0.2,
        "recall": 0.2,
        "relevance": 0.1
    }
    output_prefix = os.path.join(root_dir, "eval_results", "post_eval")
    evaluator.post_process_results(
        results=results,
        weights=weights,
        output_prefix=output_prefix
    )

if __name__ == "__main__":
    start = time.time()
    main_eval()
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
