import os
import asyncio
from glob import glob
import torch
from litellm.litellm_core_utils.litellm_logging import Logging
from sentence_transformers import SentenceTransformer
from .utils.data_utils import load_guidelines, load_transcript, segment_transcript
from .utils.embedding_utils import summarize_embed_groups_async, summarize_match_top_k_questions_async, summarize_match_top_p_questions_async
from .utils.output_utils import generate_output_from_summarized_matches_async, setup_logging, output_divider
from .utils.api_utils import summarize_question_async
from dotenv import load_dotenv
from enum import Enum
import logging


class D2DProcessor:

    class SamplingMethod(Enum):
        TOP_K = "top_k"
        TOP_P = "top_p"
        # THRESHOLD = "threshold"

    """A class to process interview transcripts using RAG-based summarization and matching."""
    def __init__(self, llm_model: str = "gpt-4o-mini", embedding_model: str = "multi-qa-mpnet-base-dot-v1",
            sampling_method: SamplingMethod = SamplingMethod.TOP_K, max_concurrent_calls: int = 10, top_k:int = 5,
                 top_p:float = 0.5, custom_extract_prompt: str = None, custom_summarize_prompt: str = None):

        """
        Initialize the D2DProcessor with model and processing configurations.

        Args:
            llm_model (str): The LLM model to use for summarization (default: "gpt-4o-mini").
            embedding_model (str): SentenceTransformer model name (default: "multi-qa-mpnet-base-dot-v1").
            max_concurrent_calls (int): Maximum concurrent API calls for async processing.
        """
        load_dotenv()
        self.llm_model = llm_model
        self.max_concurrent_calls = max_concurrent_calls
        self.sampling_method = sampling_method
        self.top_k = top_k
        self.top_p = top_p
        self.custom_extract_prompt = custom_extract_prompt
        self.custom_summarize_prompt = custom_summarize_prompt

        # Initialize SentenceTransformer model
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.embedding_model.to(self.device)

    def log_config(self, logger = logging.Logger):
        logger.info("Pipeline configuration:")
        logger.info(f"LLM Model: {self.llm_model}")
        logger.info(f"Embedding Model: {self.embedding_model_name}")
        logger.info(f"Sampling Method: {self.sampling_method.value}")
        logger.info(f"Max Concurrent Calls: {self.max_concurrent_calls}")
        logger.info(f"Top K: {self.top_k}")
        logger.info(f"Top P: {self.top_p}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Custom Extract Prompt: {'Set' if self.custom_extract_prompt else 'Not set'}")
        logger.info(f"Custom Summarize Prompt: {'Set' if self.custom_summarize_prompt else 'Not set'}")
        output_divider(logger)

    def process_transcripts(self, data_dir: str, interview_name: str, output_dir: str,
                            disable_logging: bool = False) -> None:
        """
        Process all transcripts in the directory and generate summarized matches.

        Args:
            data_dir (str): Directory containing transcript files.
            interview_name (str): The interview to be processed, e.g., "interview_1090" or "interview_abcr".
            output_dir (str): Path for the output files directory.
            pipeline_name (str): Name of the pipeline for logging.
            disable_logging (bool): Whether to disable logging.

        Output: files will be saved to the output directory(output_dir) as follows::
            matched_responses.csv: CSV file with matched responses for each transcript.
            generator_log.txt: Log file for the generator execution.
            pipeline_log.log: Log file for the pipeline execution.
        """

        transcript_dir = os.path.join(data_dir, interview_name)
        guidelines_path = os.path.join(data_dir, f"{interview_name}_guidelines.csv")

        interview_name = interview_name.split("_")[-1]
        output_path = os.path.join(output_dir, f"D2D_survey_{interview_name}.csv")

        # Run async processing within the event loop
        loop = asyncio.get_event_loop()

        if loop.is_running():
            # If already in an async context, create a new loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self._process_transcripts_async(
                    transcript_dir, guidelines_path, output_path, disable_logging
                ))
            finally:
                new_loop.close()
        else:
            # Normal case: run in current loop
            loop.run_until_complete(self._process_transcripts_async(
                transcript_dir, guidelines_path, output_path, disable_logging
            ))

    async def _process_transcripts_async(self, transcript_dir: str, guidelines_path: str, output_path: str,
                                        disable_logging: bool) -> None:
        """
        Internal async method to process transcripts.

        Args:
            transcript_dir (str): Directory containing transcript files.
            guidelines_path (str): Path to the guidelines CSV file.
            output_path (str): Path for the output CSV file.
            pipeline_name (str): Name of the pipeline for logging.
            disable_logging (bool): Whether to disable logging.
        """
        # Set up logging
        pipeline_name = "D2D"
        logger = setup_logging(pipeline_name, output_path, disable_logging=disable_logging)
        logger.info("D2D processing pipeline started...")
        output_divider(logger, True)
        self.log_config(logger)
        logger.info(
            f"Start finding matches for guidelines: {guidelines_path.split('/')[-1]} with sampling method: {self.sampling_method.value}"
        )

        # Load guidelines
        guide_questions = load_guidelines(guidelines_path)

        print(transcript_dir)

        # Find transcript files
        transcript_files = glob(os.path.join(transcript_dir, "*.txt"))
        if not transcript_files:
            logger.error("No transcript files found in directory")
            raise FileNotFoundError("No transcript files found in directory")

        # Precompute embeddings for guide questions
        logger.info("Summarizing and embedding guide questions...")
        guide_question_data = await self._summarize_guide_questions(guide_questions, logger)
        logger.info("Guide question embeddings precomputed.")
        output_divider(logger, True)

        # Process each transcript
        matches_list = []
        for transcript_path in transcript_files:
            logger.info(f"Processing transcript: {transcript_path.split('/')[-1]}")
            transcript_matches = await self._process_single_transcript(transcript_path, guide_question_data, logger)
            matches_list.append(transcript_matches)


        # Generate output
        await generate_output_from_summarized_matches_async(
            transcript_files, matches_list, guide_questions, self.llm_model, output_path,
            max_concurrent_calls=self.max_concurrent_calls, logger=logger,
            embedding_model=self.embedding_model, device=self.device,
            custom_extract_prompt=self.custom_extract_prompt,
            custom_summarize_prompt=self.custom_summarize_prompt
        )
        logger.info("Processing completed.")

    async def _summarize_guide_questions(self, guide_questions: list, logger) -> list:
        """
        Summarize and embed guide questions asynchronously.

        Args:
            guide_questions (list): List of guide questions.
            logger: Logger instance for logging.

        Returns:
            list: List of dictionaries with guide question data and embeddings.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        async def summarize_with_limit(question):
            async with semaphore:
                return await summarize_question_async(question, self.llm_model, logger)

        tasks = [summarize_with_limit(q) for q in guide_questions]
        summarized_questions = await asyncio.gather(*tasks)

        guide_question_data = []
        for guide_question, summarized_question in zip(guide_questions, summarized_questions):
            embedding = self.embedding_model.encode(summarized_question, convert_to_tensor=True, device=self.device)
            guide_question_data.append({
                "guide_question": guide_question,
                "summarized_guide_question": summarized_question,
                "embedding": embedding
            })
        return guide_question_data

    async def _process_single_transcript(self, transcript_path: str, guide_question_data: list, logger) -> list:
        """
        Process a single transcript and match it against guide questions.

        Args:
            transcript_path (str): Path to the transcript file.
            guide_question_data (list): List of guide question data with embeddings.
            logger: Logger instance for logging.

        Returns:
            list: List of matches for the transcript.
        """
        transcript = load_transcript(transcript_path)
        groups = segment_transcript(transcript)
        group_embeddings = await summarize_embed_groups_async(groups, self.embedding_model, self.device, self.llm_model, logger)

        transcript_matches = []

        if self.sampling_method == self.SamplingMethod.TOP_K:
            for guide_data in guide_question_data:
                top_k_matches = await summarize_match_top_k_questions_async(
                    guide_data["embedding"], group_embeddings, k=self.top_k
                )
                transcript_matches.append({
                    "guide_question": guide_data["guide_question"],
                    "matches": top_k_matches
                })

        if self.sampling_method == self.SamplingMethod.TOP_P:
            for guide_data in guide_question_data:
                top_matches = await summarize_match_top_p_questions_async(
                    guide_data["embedding"], group_embeddings, p=self.top_p)
                transcript_matches.append({
                    "guide_question": guide_data["guide_question"],
                    "matches": top_matches
                })

        return transcript_matches