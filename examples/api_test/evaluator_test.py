import sys
import os
import time
import argparse
import ast

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

    # Run evaluation
    results = evaluator.evaluate(
        log_input_path=log_input_path,
        context_output_path=context_output_path,
        rag_csv_path=rag_csv_path,
        ref_csv_path=ref_csv_path,
        eval_output_path=eval_output_path
    )

    # Post-processing
    evaluator.post_process_results(
        results=results,
        weights=weights,
        output_prefix=post_eval_prefix
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run D2D evaluation pipeline.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM generation")
    parser.add_argument("--max_concurrent_calls", type=int, default=5, help="Maximum concurrent LLM API calls.")
    parser.add_argument("--weights", type=str, help="Weights dictionary, e.g., '{\"correctness\": 0.3, \"faithfulness\": 0.2, \"precision\": 0.2, \"recall\": 0.2, \"relevance\": 0.1}'")

    parser.add_argument("--log_input", type=str, required=True, help="Path to generator log file")
    parser.add_argument("--context_output", type=str, required=True, help="Path to save retrieved contexts")
    parser.add_argument("--rag_csv", type=str, required=True, help="Path to RAG answers CSV")
    parser.add_argument("--ref_csv", type=str, required=True, help="Path to reference answers CSV")
    parser.add_argument("--eval_output", type=str, required=True, help="Path to evaluation result CSV")
    parser.add_argument("--post_eval_prefix", type=str, required=True, help="Prefix path for post-eval output files")

    args = parser.parse_args()

    if args.weights:
        try:
            weights = ast.literal_eval(args.weights)
            assert isinstance(weights, dict), "Weights must be a dictionary."
        except Exception as e:
            raise ValueError(f"Failed to parse --weights: {e}")
    else:
        weights = None


    start = time.time()

    main_eval(
        model=args.model,
        temperature=args.temperature,
        max_concurrent_calls=args.max_concurrent_calls,
        log_input_path=os.path.expanduser(args.log_input),
        context_output_path=os.path.expanduser(args.context_output),
        rag_csv_path=os.path.expanduser(args.rag_csv),
        ref_csv_path=os.path.expanduser(args.ref_csv),
        eval_output_path=os.path.expanduser(args.eval_output),
        post_eval_prefix=os.path.expanduser(args.post_eval_prefix),
        weights=weights
    )

    end = time.time()
    print(f"Evaluation completed in {end - start:.2f} seconds.")
