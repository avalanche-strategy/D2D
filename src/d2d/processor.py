import os
from glob import glob
import torch
from litellm import completion
from litellm.exceptions import Timeout, APIError, AuthenticationError
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from enum import Enum
import logging
import requests
import time
import json
import asyncio

from .utils.data_utils import load_guidelines, load_transcript, segment_transcript
from .utils.embedding_utils import summarize_embed_groups_async, summarize_match_top_k_questions_async, \
    summarize_match_top_p_questions_async
from .utils.output_utils import generate_output_from_summarized_matches_async, setup_logging, output_divider, \
    get_divider
from .utils.api_utils import summarize_question_async


class D2DProcessor:
    class SamplingMethod(Enum):
        """
        Enumeration of sampling methods for matching transcript groups to guide questions.
        """
        TOP_K = "top_k"
        TOP_P = "top_p"
        # THRESHOLD = "threshold"

    """A class to process interview transcripts using RAG-based summarization and matching."""

    def __init__(self, llm_model: str = "gpt-4o-mini", embedding_model: str = "multi-qa-mpnet-base-dot-v1",
                 sampling_method: SamplingMethod = SamplingMethod.TOP_K, max_concurrent_calls: int = 10, top_k: int = 5,
                 top_p: float = 0.5, thematic_alignment_similarity_threshold: float = 0.4,
                 custom_extract_prompt: str = None, custom_summarize_prompt: str = None):
        """
        Initialize the D2DProcessor with model and processing configurations.

        Args:
            llm_model (str): The LLM model to use for summarization (default: "gpt-4o-mini").
            embedding_model (str): SentenceTransformer model name (default: "multi-qa-mpnet-base-dot-v1").
            sampling_method (SamplingMethod): Method for sampling matches (default: SamplingMethod.TOP_K).
            max_concurrent_calls (int): Maximum concurrent API calls for async processing (default: 10).
            top_k (int): Number of top matches to consider when using TOP_K sampling (default: 5).
            top_p (float): Similarity threshold for TOP_P sampling (default: 0.5).
            thematic_alignment_similarity_threshold (float): Threshold for thematic alignment check (default: 0.4).
            custom_extract_prompt (str, optional): Custom prompt for extraction (default: None).
            custom_summarize_prompt (str, optional): Custom prompt for summarization (default: None).

        Raises:
            RuntimeError: If no internet connection is detected, as it’s required to download the embedding model.
        """
        # Check internet connection here because setting up embedding model requires internet connection
        if not D2DProcessor._check_internet_connection():
            print(get_divider())
            print(
                "Error: No internet connection detected. Please check your internet connection and restart the process.")
            print(get_divider())
            raise RuntimeError("Internet connection required to initialize D2DProcessor.")

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

    def _log_config(self, logger=logging.Logger):
        """
        Log the current configuration of the D2DProcessor using the provided logger.

        Args:
            logger (logging.Logger): Logger instance to use for logging.
        """
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

    @staticmethod
    def _check_internet_connection(timeout=5):
        """
        Check if the user's machine has an active internet connection.

        Tests connectivity by attempting a GET request to a reliable external server
        (e.g., google.com). This helps differentiate local network issues from LLM
        service unresponsiveness.

        Parameters
        ----------
        timeout : int, optional
            The maximum time in seconds to wait for a response (default is 5).
            A shorter timeout allows quick detection of connectivity issues.

        Returns
        -------
        bool
            True if the internet connection is active (request succeeds), False otherwise.

        Notes
        -----
        - Uses a simple HTTP GET request to 'https://www.google.com' as a connectivity test.
        - Catches requests.RequestException to handle network-related errors.
        - Does not guarantee LLM API availability; use with an LLM-specific test for
          comprehensive error handling.
        """
        try:
            requests.get('https://www.google.com', timeout=timeout)
            print("Internet connection check successful.")
            return True
        except requests.RequestException as e:
            print(f"Internet connection check failed: {str(e)}")
            return False

    def _test_llm_connection(self):
        """
        Test if the LLM is responding with a synchronous call and switch providers if needed.

        Performs a simple completion request to verify LLM availability. If the initial model fails,
        it attempts the default model for the other provider (OpenAI or Anthropic). Uses defaults
        from llm_defaults.json.

        Returns
        -------
        bool
            True if the LLM responds successfully, False otherwise.

        Notes
        -----
        - Uses a minimal prompt ("Test") to check responsiveness.
        - Implements up to 3 retries per model with exponential backoff (5, 10 seconds).
        - Skips waiting on the last attempt.
        - Switches between OpenAI and Anthropic if the initial provider fails.
        - Logs specific error messages for API key issues or LLM unresponsiveness.
        """
        import os, json, time
        from litellm import completion
        from litellm.exceptions import Timeout, APIError, AuthenticationError

        config_path = os.path.join(os.path.dirname(__file__), 'config', 'llm_defaults.json')
        default_models = {}
        try:
            with open(config_path, 'r') as f:
                default_models = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found. Using defaults.")
            default_models = {"openai": "gpt-4o-mini", "anthropic": "claude-3-5-sonnet"}
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_path}. Using defaults.")
            default_models = {"openai": "gpt-4o-mini", "anthropic": "claude-3-5-sonnet"}

        initial_model = self.llm_model
        provider = "openai" if "gpt" in initial_model.lower() else "anthropic" if "claude" in initial_model.lower() else "openai"
        fallback_provider = "anthropic" if provider == "openai" else "openai"
        fallback_model = default_models[fallback_provider]

        max_retries = 3
        base_delay = 5  # seconds
        current_model = initial_model

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=current_model,
                    messages=[{"role": "user", "content": "Test"}],
                    temperature=0,
                    timeout=10
                )
                if response.choices and response.choices[0].message.content:
                    print(f"LLM connection test successful with {current_model}.")
                    self.llm_model = current_model  # Update if switched
                    return True
            except AuthenticationError as e:
                print(
                    f"API Key Error with {current_model}: {str(e)}. Please check your API key for {provider.upper()} in .env.")
                return False
            except APIError as e:
                if hasattr(e, 'status_code'):
                    if e.status_code in [503, 429]:  # Service unavailable or rate limit
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(
                                f"LLM Unresponsive (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Retrying in {delay} seconds...")
                            time.sleep(delay)
                        else:
                            print(
                                f"LLM Unresponsive (final attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. No more retries.")
                    else:
                        print(
                            f"API Error (attempt with {current_model}): {str(e)}. Check configuration.")
                        return False
            except Timeout as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"LLM Timeout (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(
                        f"LLM Timeout (final attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. No more retries.")
            except Exception as e:
                print(
                    f"Unexpected error during LLM test with {current_model}: {str(e)}.")
                return False

        # Switch to fallback provider if initial model fails after retries
        print(f"Switching to {fallback_model} from {fallback_provider} after {initial_model} failed.")
        current_model = fallback_model
        for attempt in range(max_retries):
            try:
                response = completion(
                    model=current_model,
                    messages=[{"role": "user", "content": "Test"}],
                    temperature=0,
                    timeout=10
                )
                if response.choices and response.choices[0].message.content:
                    print(f"LLM connection test successful with {current_model}.")
                    self.llm_model = current_model  # Update to fallback model
                    return True
            except AuthenticationError as e:
                print(
                    f"API Key Error (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Please check your API key for {fallback_provider.upper()} in .env.")
                return False
            except APIError as e:
                if hasattr(e, 'status_code'):
                    if e.status_code in [503, 429]:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(
                                f"LLM Unresponsive (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Retrying in {delay} seconds...")
                            time.sleep(delay)
                        else:
                            print(
                                f"LLM Unresponsive (final attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. No more retries.")
                    else:
                        print(
                            f"API Error (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Check configuration.")
                        return False
            except Timeout as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"LLM Timeout (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(
                        f"LLM Timeout (final attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}. No more retries.")
            except Exception as e:
                print(
                    f"Unexpected error during LLM test (attempt {attempt + 1}/{max_retries} with {current_model}): {str(e)}.")
                return False

        print(
            f"LLM connection test failed for both {initial_model} and {fallback_model}. Check API keys or service status.")
        return False
    
    def process_transcripts(self, transcripts_dir: str, guidelines_path: str, interview_name: str, output_dir: str,
                            disable_logging_to_console: bool = True) -> None:
        """
        Process all transcripts in the directory and generate a single CSV file with summarized matches for all transcripts.

        Args:
            transcripts_dir (str): Directory containing transcript files.
            guidelines_path (str): Path to the guidelines CSV file.
            interview_name (str): The interview identifier, used in the output file name.
            output_dir (str): Directory to save the output CSV file.
            disable_logging_to_console (bool): Whether to disable logging to console (default: True).

        Output:
            A CSV file named "D2D_survey_{interview_name}.csv" will be saved in the output directory,
            containing the summarized matches for all processed transcripts.

        Raises:
            RuntimeError: If the LLM connection test fails.
        """



        # Check LLM connection
        if not self._test_llm_connection():
            print(get_divider())
            print(
                "Error during LLM connection test.")
            print(get_divider())
            raise RuntimeError("LLM connection test failed. See logs for details.")

        # User friendly output to console
        print(get_divider())
        print(
            f"Processing transcripts in \"{transcripts_dir}\" \nfor guidelines {guidelines_path} \nand saving output to \"{output_dir}\" ...")
        print(get_divider())

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
                    transcripts_dir, guidelines_path, output_path, disable_logging_to_console
                ))
            finally:
                new_loop.close()
        else:
            # Normal case: run in current loop
            loop.run_until_complete(self._process_transcripts_async(
                transcripts_dir, guidelines_path, output_path, disable_logging_to_console
            ))

    async def _process_transcripts_async(self, transcript_dir: str, guidelines_path: str, output_path: str,
                                         disable_logging_to_console: bool) -> None:
        """
        Internal async method to process transcripts and generate the output CSV.

        Args:
            transcript_dir (str): Directory containing transcript files.
            guidelines_path (str): Path to the guidelines CSV file.
            output_path (str): Path to save the output CSV file.
            disable_logging_to_console (bool): Whether to disable logging to console.
        """

        # Set up logging
        pipeline_name = "D2D"
        logger = setup_logging(pipeline_name, output_path, disable_logging_to_console=disable_logging_to_console)
        logger.info("D2D processing pipeline started...")
        output_divider(logger, True)
        self._log_config(logger)
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

        try:
            guide_question_data = await self._summarize_guide_questions(guide_questions, logger)
            logger.info("Guide question embeddings precomputed.")
        except Exception as e:
            logger.error(f"Error computing Guide question embeddings {e}")
            raise e
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
        Summarize and embed guide questions asynchronously using the LLM.

        Args:
            guide_questions (list): List of guide questions to summarize and embed.
            logger (logging.Logger): Logger instance for logging.

        Returns:
            list: List of dictionaries, each containing the original guide question, its summarized version, and its embedding.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        async def summarize_with_limit(question):
            async with semaphore:
                return await summarize_question_async(question, self.llm_model, logger)

        tasks = [summarize_with_limit(q) for q in guide_questions]
        summarized_questions = await asyncio.gather(*tasks, return_exceptions=True)

        # check for task errors
        error_results = [e for e in summarized_questions if isinstance(e, Exception)]
        good_results = [r for r in summarized_questions if not isinstance(r, Exception)]

        if len(good_results) == 0 and len(error_results) > 0:
            error_list = set([e.__class__.__qualname__ for e in error_results])
            logger.error(f"All Guide Question Summarizing tasks returned errors of type: {error_list}")
            raise error_results[0]
        elif len(error_results) > 0:
            logger.warning("Some Guide Question Summarizing tasks have errors.")

        guide_question_data = []
        for guide_question, summarized_question in zip(guide_questions, summarized_questions):
            # Fallback to original question if this instance resulted in an error
            if isinstance(summarized_question, Exception):
                logger.error(f"Error summarizing guide question '{guide_question}': {str(summarized_question)}")
                summarized_question = guide_question

            embedding = self.embedding_model.encode(summarized_question, convert_to_tensor=True, device=self.device)
            guide_question_data.append({
                "guide_question": guide_question,
                "summarized_guide_question": summarized_question,
                "embedding": embedding
            })
        return guide_question_data

    async def _thematic_alignment_check(self, group_embeddings: list, guide_question_data: list, transcript_path: str,
                                        logger: logging.Logger) -> bool:
        """
        Check if the transcript is thematically aligned with the guideline questions based on cosine similarity.

        If the similarity is below the threshold, prompt the user to decide whether to proceed.

        Args:
            group_embeddings (list): List of transcript group embeddings and metadata.
            guide_question_data (list): List of guide question data with embeddings.
            transcript_path (str): Path to the transcript file for display in the user prompt.
            logger (logging.Logger): Logger instance for logging.

        Returns:
            bool: True if the transcript is aligned or the user chooses to proceed, False if the user chooses to skip.
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

    async def _process_single_transcript(self, transcript_path: str, guide_question_data: list,
                                         logger: logging.Logger) -> list:
        """
        Process a single transcript by segmenting it, summarizing and embedding its content,
        and matching it against the guide questions based on the sampling method.

        Args:
            transcript_path (str): Path to the transcript file.
            guide_question_data (list): List of guide question data with embeddings.
            logger (logging.Logger): Logger instance for logging.

        Returns:
            list: List of dictionaries, each containing a guide question and its matched transcript groups.
        """

        # Load the transcript file into a structured format
        transcript = load_transcript(transcript_path)

        # Segment the transcript into groups of interviewer-interviewee exchanges
        # Each group contains interviewer questions, interviewee responses, line references, and speaking round
        groups = segment_transcript(transcript)

        # Summarize and embed the interviewer questions and interviewee responses for each group
        # Uses the LLM to summarize questions and SentenceTransformer to generate embeddings
        try:
            group_embeddings = await summarize_embed_groups_async(
                groups, self.embedding_model, self.device, self.llm_model, logger
            )
        except Exception as e:
            logger.info(f"Error embedding file {transcript_path.split('/')[-1]} due to {e}")
            return []

        # Check thematic alignment between transcript and guideline
        logger.info("Checking thematic alignment between transcript and guideline...")
        should_proceed = await self._thematic_alignment_check(group_embeddings, guide_question_data, transcript_path,
                                                              logger)

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