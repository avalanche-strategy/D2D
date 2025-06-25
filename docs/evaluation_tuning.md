## Adjusting the Similarity Threshold for Empty or Ambiguous Answers

In the evaluation pipeline, if both the generated answer and the reference answer are vague, empty, or ambiguous, we should avoid penalizing the model with a low correctness or faithfulness score. Instead, we treat this as a **valid match**, indicating that the model appropriately responded with uncertainty when no answer was available.

This behavior is implemented in the function `is_answer_empty_or_confused()`.

### Function Location  
This function is defined in `d2d/utils/eval_config_utils.py`, used in `d2d/evaluation/eval_prompt_utils.py`, and called in `d2d/evaluation/ragas_eval.py` inside the `score_ragas()` function.

### Working Mechanism  
This function uses **embedding similarity** to compare a given answer against a list of pre-defined vague/confused templates such as:
- "I don't know"
- "No idea"
- "This is unclear"
- "There is no information"
- "N/A"

### Set the Similarity Threshold
In `d2d/utils/eval_config_utils.py`, the similarity threshold for detecting vague or ambiguous answers is defined as:
```python
SIMILARITY_THRESHOLD = 0.78
```
This constant controls how strictly we classify an answer as "empty", "confused", or "non-informative".

### Suggested Values

We recommend tuning the similarity threshold based on your use case. Below is a guideline for selecting appropriate values:

| Threshold      | Behavior                                      |
|----------------|-----------------------------------------------|
| `> 0.85`       | **Very strict**: fewer false positives, but might miss genuinely vague answers |
| `0.75–0.80`    | **Balanced** (default = 0.78): good trade-off between sensitivity and specificity |
| `< 0.70`       | **Very lenient**: more answers flagged as vague, may misclassify meaningful responses |

---

## Modifying Evaluation Prompts

Prompts sent to the LLM for scoring are constructed by `build_prompt(metric, row, chunk_list)`.

### Function Location  
This function is defined in `d2d/utils/eval_prompt_utils.py`, and called in `d2d/evaluation/ragas_eval.py` by `score_ragas()` function within the scoring loop per metric.

### Adjust the Prompt
You can modify the template inside `build_prompt()` to:
- Change the wording
- Provide more explicit scoring instructions
- Add few-shot examples for more robust scoring behavior
- Emphasize precision/recall/factual constraints

Prompt example:
```python
prompt = f"""
You are evaluating the {metric} of a generated answer.
Question: {question}
Reference Answer: {answer_ref}
Generated Answer: {answer_rag}
Context: {context}  # Optional
Please assign a score between 1 and 5 and explain your reasoning.
"""
```

Tip: Keep temperature = 0 for deterministic scoring.