# Dialogue2Data (D2D) User Guide

Welcome to the technical user guide for `d2d` (Dialogue2Data).  
D2D is a Python-based, open-source toolkit for converting unstructured interview transcripts into structured, analyzable data using advanced NLP and LLM techniques. The package streamlines the extraction, matching, and summarization of responses based on your predefined guideline questions—ideal for researchers, analysts, and organizations seeking efficient insights from qualitative data.

This guide provides detailed information about the package’s classes, parameters, and pipeline configuration. For general usage, installation, or a quick start, please refer to the [README](https://github.com/avalanche-strategy/D2D/blob/dom/final_report_cut_words/README.md).

---

## `D2DProcessor` Initialization Parameters

> **Note:** All parameters have default settings. In most cases, you can simply use:
> ```python
> processor = D2DProcessor()
> ```
> See below for advanced customization.

- **llm_model : str**  
  The name of the Large Language Model to use for summarization (default: `"gpt-4o-mini"`).  
  *Requires a valid API key set in your `.env` file. See Additional Notes.*

- **embedding_model : str**  
  The name of the SentenceTransformer embedding model (default: `"multi-qa-mpnet-base-dot-v1"`).  
  *Downloads from HuggingFace if not already present. Requires internet on first use.*

- **sampling_method : D2DProcessor.SamplingMethod**  
  Strategy for selecting transcript segments for each guideline question.  
  - `SamplingMethod.TOP_K`: Selects the top K most similar segments (default).
  - `SamplingMethod.TOP_P`: Selects segments until cumulative similarity probability reaches P.

- **max_concurrent_calls : int**  
  Maximum number of concurrent API calls to the LLM or embedding model (default: `10`).  
  *Tune for your hardware or LLM provider’s rate limits.*

- **top_k : int, optional**  
  Number of top segments to select per guideline question when using `TOP_K` (default: `5`).  
  *Ignored if using `TOP_P`.*

- **top_p : float, optional**  
  Similarity threshold (cumulative probability) for `TOP_P` sampling (default: `0.5`).  
  *Ignored if using `TOP_K`.*

- **thematic_alignment_similarity_threshold : float, optional**  
  Minimum cosine similarity for thematic alignment between transcript and guidelines (default: `0.4`).  
  *If similarity is lower, the user is prompted whether to skip the transcript.*

- **custom_extract_prompt : str, optional**  
  Custom prompt template for LLM extraction of key phrases.  
  Use `{context}` for the transcript and `{query}` for the guideline question.  
  If not set, a robust default prompt is used.

- **custom_summarize_prompt : str, optional**  
  Custom prompt template for summarizing extracted phrases.  
  Use `{extracted_phrase}` and `{query}` placeholders.  
  If not set, a robust default prompt is used.

---

## `D2DEvaluator` Initialization Parameters 

> All parameters have default values—so you can usually just run:  
> ```python
> evaluator = D2DEvaluator()
> ```
> See below for customization options.

- **model : str**  
  The name of the LLM model used for evaluation (e.g., `"gpt-4o-mini"`). This model scores the generated outputs across various evaluation metrics such as faithfulness and relevance.

- **temperature : float**  
  The decoding temperature for the LLM. Lower values (e.g., `0.0`) result in more deterministic outputs, while higher values introduce more randomness.

- **max_concurrent_calls : int**  
  The maximum number of concurrent API requests allowed when querying the LLM. This setting helps balance evaluation speed and system/API limits during batch processing.

## Evaluation Parameters `evaluate`

- **log_input_path : str**  
  Path to the generator log file that contains the raw logs from which retrieved context blocks will be extracted.

- **context_output_path : str**  
  File path where the extracted context DataFrame will be saved as a CSV. This file is later used for evaluation.

- **rag_csv_path : str**  
  Path to the CSV file containing model-generated answers. These are the answers being evaluated.

- **ref_csv_path : str**  
  Path to the CSV file containing the human-annotated reference answers. These serve as the ground truth for comparison.

- **eval_output_path : str**  
  File path where the final evaluation results (with metric scores) will be saved as a CSV.

## Post-Evaluation Parameters `post_process_results`

- **results : pd.DataFrame**  
  A DataFrame containing the full evaluation output, including metric scores for each response (e.g., faithfulness, correctness).

- **weights : dict**  
  A dictionary specifying the weight assigned to each metric (e.g., `{"faithfulness": 0.3, "relevance": 0.2, ...}`). The weights must sum to 1.0 and are used to compute a joint performance score.

- **output_prefix : str, optional**  
  Prefix used when saving the post-processed output files. Two CSVs will be saved:
  - `<output_prefix>_highlighted.csv`: Rows where any metric score is below the threshold.
  - `<output_prefix>_joint_metric.csv`: Respondent-level joint scores based on weighted averages.  
  Default is `"post_eval"`.

- **highlight_threshold : float, optional**  
  Threshold below which a metric score is considered low. Scores equal to or under this value are flagged and highlighted in the output. Default is `1.0`.

---

## Additional Notes

- **Internet Required:** Initialization requires an active internet connection for model downloads and LLM access.
- **API Keys:** Ensure valid API keys are set via `.env` for LLM providers (OpenAI, Anthropic). The processor logs and halts if authentication fails.
- **Automatic Device Selection:** Embedding models are loaded to the optimal device (MPS, CUDA, or CPU) based on your system.
- **Interactive Workflow:** If a transcript is **not thematically aligned with** guidelines, you will be prompted to continue or skip processing for that file.
- **Async Processing:** The pipeline leverages asynchronous processing for faster throughput with large datasets.
- **Robust LLM Handling:** LLM connectivity is tested and will fall back to alternative providers/models as needed.
- **Logging:** Detailed logs are created by default for debugging and review; console logging can be disabled if desired.
- **File Formats:** Input transcripts must be `.txt` and guidelines `.csv`; all output is saved in the specified output directory.
- **User Input:** Some processing steps require interactive confirmation; for automated workflows, review the code or set the similarity threshold accordingly.
- **Batch Size:** Adjust `max_concurrent_calls` for your compute/API quota; excessive values may result in API throttling or local issues.
