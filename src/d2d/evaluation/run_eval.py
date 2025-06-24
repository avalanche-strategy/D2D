import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from src.utils.log_utils import extract_retrieved_contexts
from src.evaluation.ragas_eval import run_ragas_evaluation

def main():
    """
    Run a two-step pipeline: extract retrieved contexts from logs and evaluate answers using GPT-based metrics.
    
    Steps:
    1. Extract retrieved contexts from a log file and save as CSV.
    2. Run RAGAS evaluation on provided RAG answers, references, and extracted context.
    """
    parser = argparse.ArgumentParser(description="Run full pipeline: extract context + GPT evaluation")
    parser.add_argument("--log_input_path", required=True, help="Path to input .log file")
    parser.add_argument("--context_output_path", required=True, help="Path to save extracted retrieved_contexts CSV")
    parser.add_argument("--rag", required=True, help="Path to RAG-generated answers CSV")
    parser.add_argument("--ref", required=True, help="Path to reference answers CSV")
    parser.add_argument("--eval_output_path", required=True, help="Path to save evaluation results CSV")

    args = parser.parse_args()

    # Step 1: Extract context from log
    print("Extracting retrieved contexts from log...")
    extract_retrieved_contexts(args.log_input_path, args.context_output_path)
    print(f"Context extraction complete: {args.context_output_path}")

    # Step 2: Run RAGAS evaluation
    print("Running evaluation...")
    run_ragas_evaluation(
        rag_path=args.rag,
        ref_path=args.ref,
        context_path=args.context_output_path,
        output_path=args.eval_output_path
    )

if __name__ == "__main__":
    main()
