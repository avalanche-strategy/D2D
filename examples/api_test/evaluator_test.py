import sys
import os
import time
import glob

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

# Automatically finds most recent log and response files
def find_latest_files(base_dir: str, topic: str):
    log_pattern = os.path.join(base_dir, f"D2D_survey_{topic}_generator_log_*.txt")
    rag_pattern = os.path.join(base_dir, f"D2D_survey_{topic}_responses_*.csv")

    log_files = glob.glob(log_pattern)
    rag_files = glob.glob(rag_pattern)

    if not log_files:
        raise FileNotFoundError(f"No log files found for {topic}")
    if not rag_files:
        raise FileNotFoundError(f"No response files found for {topic}")

    latest_log = max(log_files, key=os.path.getmtime)
    latest_rag = max(rag_files, key=os.path.getmtime)

    return latest_log, latest_rag

# Default test setup
def main_default_eval():
    topic = "food"
    base_dir = "results"
    eval_dir = "eval_results"
    ref_dir = "data/synthetic_data"

    model = "gpt-4o-mini"
    temperature = 0.0
    max_concurrent_calls = 5
    weights = None

    log_input, rag_csv = find_latest_files(base_dir, topic)
    context_output = os.path.join(eval_dir, f"retrieved_contexts_{topic}.csv")
    ref_csv = os.path.join(ref_dir, f"interview_{topic}_guidelines.csv")
    eval_output = os.path.join(eval_dir, f"eval_output_{topic}.csv")
    post_eval_prefix = os.path.join(eval_dir, f"eval_output_post_{topic}")

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