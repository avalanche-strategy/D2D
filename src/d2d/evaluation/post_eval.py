import argparse
import pandas as pd

METRICS = ["faithfulness", "correctness", "precision", "recall", "relevance"]

def highlight_low_scores(df: pd.DataFrame, lesseq: float, output_path: str):
    """
    Highlight rows where any metric score is less than or equal to a threshold.

    For each matched score, replaces the value with a 'low-<score>' tag.
    Saves the filtered rows to a CSV file.

    Parameters:
    - df (pd.DataFrame): Input evaluation DataFrame.
    - lesseq (float): Threshold for flagging low scores.
    - output_path (str): Path to save the highlighted CSV.
    """
    highlight_rows = []

    for idx, row in df.iterrows():
        row_copy = row.copy()
        found = False
        for metric in METRICS:
            score_col = f"{metric}_score"
            score_val = row_copy.get(score_col)
            if pd.notna(score_val) and score_val <= lesseq:
                row_copy[score_col] = f"low-{score_val}"
                found = True
        if found:
            highlight_rows.append(row_copy)

    highlight_df = pd.DataFrame(highlight_rows)
    highlight_df.to_csv(output_path, index=False)
    print(f"Highlighted rows saved to: {output_path}")


def compute_joint_metric(df: pd.DataFrame, weights: dict, output_path: str):
    """
    Compute joint metric scores per respondent using weighted averages.

    Groups scores by 'respondent_id', averages the individual metrics,
    computes a joint score using user-defined weights, and saves the result.

    Parameters:
    - df (pd.DataFrame): Input evaluation DataFrame.
    - weights (dict): Dictionary mapping metric names to weights. Sum must be 1.0.
    - output_path (str): Path to save the summary CSV.
    """
    for metric in METRICS:
        df[metric + "_score"] = pd.to_numeric(df[metric + "_score"], errors="coerce")

    group = df.groupby("respondent_id")[ [f"{m}_score" for m in METRICS] ].mean().reset_index()

    # Validate weights
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"The sum of weights must be exactly 1.0. Currently: {total_weight}")

    # Compute joint score
    group["joint_score"] = sum(group[f"{m}_score"] * weights.get(m, 0.0) for m in METRICS)

    # Sort by joint_score descending
    group = group.sort_values(by="joint_score", ascending=False)

    # Save with 1 decimal place
    group.to_csv(output_path, index=False, float_format="%.1f")
    print(f"Joint metric summary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process GPT evaluation results.")
    parser.add_argument("--input", required=True, help="Path to GPT evaluation result CSV")
    parser.add_argument("--highlight", type=float, help="Highlight rows where any score <= N (can be decimal)")
    parser.add_argument("--joint", action="store_true", help="Enable joint metric computation")
    parser.add_argument("--out_prefix", default="post_eval", help="Prefix for output CSV files")

    for m in METRICS:
        parser.add_argument(f"--{m}", type=float, default=0.2, help=f"Weight for {m} in joint score")

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    # Validate highlight threshold
    if args.highlight is not None:
        if not isinstance(args.highlight, (float, int)):
            raise ValueError(f"highlight value must be a number. Got: {args.highlight}")
        if args.highlight < 0 or args.highlight > 5:
            raise ValueError(f"highlight value must be between 0 and 5. Got: {args.highlight}")

        output_path = f"{args.out_prefix}_highlighted.csv"
        highlight_low_scores(df, args.highlight, output_path)

    # Validate joint weights and compute
    if args.joint:
        weights = {m: getattr(args, m) for m in METRICS}
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"The sum of weights must be exactly 1.0. Currently: {total_weight}")

        output_path = f"{args.out_prefix}_joint_metric.csv"
        compute_joint_metric(df, weights, output_path)
