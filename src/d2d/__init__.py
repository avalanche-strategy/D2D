"""
The d2d package provides tools for processing and evaluating interview transcripts.

This package includes utilities for loading, segmenting, embedding, and summarizing transcript data,
as well as matching responses to guideline questions using large language models (LLMs) and
sentence embeddings. The main classes are:

- D2DProcessor: Handles transcript processing, including question matching, response extraction,
  and summarization.
- D2DEvaluator: Evaluates the quality of processed outputs against expected results.

For detailed usage examples, see `D2D/docs/example.ipynb`. Ensure a `.env` file is configured with
necessary API keys (e.g., for LLM access) before using the package.

Modules:
    processor: Contains the D2DProcessor class for transcript processing.
    evaluator: Contains the D2DEvaluator class for output evaluation.
    api_utils: Utilities for LLM-based extraction and summarization.
    data_utils: Functions for loading and segmenting transcript and guideline data.
    embedding_utils: Tools for embedding and matching questions and responses.
    output_utils: Functions for logging and generating output files (CSV, JSON, log).
"""


from dotenv import load_dotenv

# Make sure it loads the .env file if it's not from the current working directory
load_dotenv()

from .processor import D2DProcessor
from .evaluator import D2DEvaluator

__all__ = ["D2DProcessor", "D2DEvaluator"]

def suppress_pydantic_serializer_warnings():
    """
    Suppress specific Pydantic serializer warnings related to unexpected values during serialization.

    This filter ignores UserWarnings emitted by Pydantic's main module that start with
    "Pydantic serializer warnings:". These warnings typically occur when non-Pydantic
    objects (e.g., LiteLLM response types) are passed to Pydantic's serialization methods.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Pydantic serializer warnings:",
        category=UserWarning,
        module="pydantic.main"
    )
suppress_pydantic_serializer_warnings()