import os
import asyncio
from glob import glob
import torch
from litellm.litellm_core_utils.litellm_logging import Logging
from sentence_transformers import SentenceTransformer
from .utils.data_utils import load_guidelines, load_transcript, segment_transcript
from .utils.embedding_utils import summarize_embed_groups_async, summarize_match_top_k_questions_async, summarize_match_top_p_questions_async
from .utils.output_utils import generate_output_from_summarized_matches_async, setup_logging, output_divider, get_divider
from .utils.api_utils import summarize_question_async
from sentence_transformers import SentenceTransformer, util
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
                 top_p:float = 0.5, thematic_alignment_similarity_threshold:float = 0.4, custom_extract_prompt: str = None,
                 custom_summarize_prompt: str = None):

        """
        Initialize the D2DProcessor with model and processing configurations.

        Args:
            llm_model (str): The LLM model to use for summarization (default: "gpt-4o-mini").
            embedding_model (str): SentenceTransformer model name (default: "multi-qa-mpnet-base-dot-v1").
            max_concurrent_calls (int): Maximum concurrent API calls for async processing.
        """

        self.llm_model = llm_model
        self.max_concurrent_calls = max_concurrent_calls
        self.sampling_method = sampling_method
        self.top_k = top_k
        self.top_p = top_p
        self.custom_extract_prompt = custom_extract_prompt
        self.custom_summarize_prompt = custom_summarize_prompt
        self.thematic_alignment_similarity_threshold = thematic_alignment_similarity_threshold

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
                            disable_logging_to_console: bool = True) -> None:
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

        # Userfriendly output to console
        print(get_divider())
        print(f"Processing transcripts for interview: \"{interview_name}\" \nin \"{data_dir}\" \nand saving to \"{output_dir}\" ...")
        print(get_divider())

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
                    transcript_dir, guidelines_path, output_path, disable_logging_to_console
                ))
            finally:
                new_loop.close()
        else:
            # Normal case: run in current loop
            loop.run_until_complete(self._process_transcripts_async(
                transcript_dir, guidelines_path, output_path, disable_logging_to_console
            ))

    async def _process_transcripts_async(self, transcript_dir: str, guidelines_path: str, output_path: str,
                                         disable_logging_to_console: bool) -> None:
        """
        Internal async method to process transcripts.

        Args:
            transcript_dir (str): Directory containing transcript files.
            guidelines_path (str): Path to the guidelines CSV file.
            output_path (str): Path for the output CSV file.
            pipeline_name (str): Name of the pipeline for logging.
            disable_logging_to_console (bool): Whether to disable logging.
        """
        # Set up logging
        pipeline_name = "D2D"
        logger = setup_logging(pipeline_name, output_path, disable_logging_to_console=disable_logging_to_console)
        logger.info("D2D processing pipeline started...")
        output_divider(logger, True)
        self.log_config(logger)
        logger.info(
            f"Start finding matches for guidelines: {guidelines_path.split('/')[-1]} with sampling method: {self.sampling_method.value}"
        )

        # Load guidelines
        guide_questions = load_guidelines(guidelines_path)

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

        # Process each transcript and filter out skipped ones
        matches_list = []
        filtered_transcript_files = []

        for transcript_path in transcript_files:
            logger.info(f"Processing transcript: {transcript_path.split('/')[-1]}")
            print(f"Processing transcript: {transcript_path.split('/')[-1]}")
            transcript_matches = await self._process_single_transcript(transcript_path, guide_question_data, logger)
            # Only include transcripts with non-empty matches (i.e., not skipped by user)
            if transcript_matches:
                matches_list.append(transcript_matches)
                filtered_transcript_files.append(transcript_path)
            else:
                logger.info(
                    f"Excluded transcript {os.path.basename(transcript_path)} from output due to empty matches (user skipped).")

        # Generate output only for non-skipped transcripts
        if matches_list:
            await generate_output_from_summarized_matches_async(
                filtered_transcript_files, matches_list, guide_questions, self.llm_model, output_path,
                max_concurrent_calls=self.max_concurrent_calls, logger=logger,
                embedding_model=self.embedding_model, device=self.device,
                custom_extract_prompt=self.custom_extract_prompt,
                custom_summarize_prompt=self.custom_summarize_prompt
            )
            logger.info("Processing completed.")
        else:
            logger.info("No transcripts to process (all skipped). No output generated.")


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

    async def _thematic_alignment_check(self, group_embeddings: list, guide_question_data: list, transcript_path: str, logger: logging.Logger) -> bool:
        """
        Check thematic alignment between transcript and guideline questions using cosine similarity.

        Args:
            group_embeddings (list): List of transcript group embeddings and metadata.
            guide_question_data (list): List of guide question data with embeddings.
            transcript_path (str): Path to the transcript file for display in prompt.
            similarity_threshold (float): Threshold for acceptable thematic similarity (default: 0.3).

        Returns:
            bool: True if processing should proceed (aligned or user agrees), False if transcript should be skipped.
        """
        # Aggregate transcript content by concatenating summarized questions
        transcript_text = " ".join(group["summarized_question"] for group in group_embeddings)
        transcript_embedding = self.embedding_model.encode(transcript_text, convert_to_tensor=True, device=self.device)

        # Aggregate guideline questions by averaging their embeddings
        guideline_embeddings = torch.stack([guide_data["embedding"] for guide_data in guide_question_data])
        guideline_embedding = torch.mean(guideline_embeddings, dim=0)

        # Compute cosine similarity between transcript and guideline embeddings
        similarity = util.cos_sim(transcript_embedding, guideline_embedding).cpu().numpy()[0][0]

        logger.info(f"Thematic similarity between transcript and guideline questions: {similarity:.3f}")

        # Check if similarity is below threshold
        if similarity < self.thematic_alignment_similarity_threshold:
            # Prompt user in console for decision
            prompt = (
                f"The transcript '{transcript_path.split('/')[-1]}' seems to deviate a lot from the guideline questions. "
                f"Similarity ({similarity:.3f}) is below threshold ({self.thematic_alignment_similarity_threshold}). \nDo you still want to proceed with this transcript? (y/n): ")
            user_input = input(prompt).strip().lower()
            # Return True if user enters 'y' or 'Y', False otherwise
            return user_input in ['y', 'yes']

        # Similarity is sufficient, proceed without prompting
        return True

    async def _process_single_transcript(self, transcript_path: str, guide_question_data: list, logger: logging.Logger) -> list:
        """
        Process a single transcript and match it against guide questions.

        Args:
            transcript_path (str): Path to the transcript file.
            guide_question_data (list): List of guide question data with embeddings.
            logger: Logger instance for logging.

        Returns:
            list: List of matches for the transcript.
        """

        # Load the transcript file into a structured format
        transcript = load_transcript(transcript_path)

        # Segment the transcript into groups of interviewer-interviewee exchanges
        # Each group contains interviewer questions, interviewee responses, line references, and speaking round
        groups = segment_transcript(transcript)

        # Summarize and embed the interviewer questions and interviewee responses for each group
        # Uses the LLM to summarize questions and SentenceTransformer to generate embeddings
        group_embeddings = await summarize_embed_groups_async(
            groups, self.embedding_model, self.device, self.llm_model, logger
        )

        # Check thematic alignment between transcript and guideline
        logger.info("Checking thematic alignment between transcript and guideline...")
        should_proceed = await self._thematic_alignment_check(group_embeddings, guide_question_data, transcript_path, logger)

        # Log user decision and handle mismatch
        if not should_proceed:
            logger.info(
                f"Skipping transcript {transcript_path.split('/')[-1]} due to user decision or thematic mismatch.")
            return []
        else:
            logger.info("Thematic alignment check passed or user chose to proceed. Continuing with matching.")

        # Initialize a list to store matches for each guide question
        transcript_matches = []

        # Process matches based on the sampling method (TOP_K or TOP_P)
        if self.sampling_method == self.SamplingMethod.TOP_K:
            # Iterate over each guide question to find the top k most similar transcript groups
            for guide_data in guide_question_data:
                # Compute cosine similarity between the guide question embedding and group embeddings
                # Return the top k matches (default k=5) with highest similarity
                top_k_matches = await summarize_match_top_k_questions_async(
                    guide_data["embedding"], group_embeddings, k=self.top_k
                )
                # Store the guide question and its matches
                transcript_matches.append({
                    "guide_question": guide_data["guide_question"],
                    "matches": top_k_matches
                })

        if self.sampling_method == self.SamplingMethod.TOP_P:
            # Iterate over each guide question to find transcript groups exceeding similarity threshold
            for guide_data in guide_question_data:
                # Compute cosine similarity and return groups with similarity >= p (default p=0.5)
                # Limits to a maximum of 5 matches, preserving speaking order
                top_matches = await summarize_match_top_p_questions_async(
                    guide_data["embedding"], group_embeddings, p=self.top_p)
                # Store the guide question and its matches
                transcript_matches.append({
                    "guide_question": guide_data["guide_question"],
                    "matches": top_matches
                })

        # Return the list of matches for all guide questions
        return transcript_matches