import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
from src.utils.log_utils import extract_retrieved_contexts


def main():
    parser = argparse.ArgumentParser(description="Extract retrieved contexts from a log file.")
    parser.add_argument(
        "--log_input_path",
        required=True,
        help="Path to the .log file to extract from."
    )
    parser.add_argument(
        "--log_output_path",
        required=True,
        help="Path to save the extracted retrieved_contexts.csv file."
    )

    args = parser.parse_args()

    df = extract_retrieved_contexts(
        log_path=args.log_input_path,
        save_path=args.log_output_path
    )

    print(f"Extraction complete. Output saved to: {args.log_output_path}")


if __name__ == "__main__":
    main()
