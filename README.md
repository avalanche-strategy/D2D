# Dialogue2Data (D2D)

## About
Dialogue2Data (D2D) is an open-source Python package that transforms unstructured interview transcripts into structured data for analysis. It consists of two major components: the **Processor**, which leverages natural language processing (NLP), large language models (LLMs), and sentence embeddings to automate topic matching, response extraction, and summarization based on discussion guides, generating structured outputs (e.g., CSV, JSON); and the **Evaluator**, which assesses output quality using metrics like faithfulness, correctness, precision, recall, and relevance. D2D is ideal for researchers and analysts processing qualitative interview data.


## Installation
1. Clone the repository:
```bash
git clone https://github.com/avalanche-strategy/D2D.git
cd D2D
```
2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate d2d
```



## Environment Configuration
To use the OpenAI and Anthropic APIs, you need to set up an environment variable for your API key. Create a `.env` file in the root directory of the project with the following content:

- **Example:**  
```bash
OPENAI_API_KEY=sk-abc123XYZ789pqr456STU012vwx789YZ
ANTHROPIC_API_KEY=sk-ant-987ZYX654WVU321TSR098qwe456PLM
```

**Note: These are fictional keys. To ensure smooth operation, please use your own API keys**



### Data Placement

To ensure smooth operation, please organize your data as follows:

**Note: Due to confidentiality, the data used in this repository, including the examples below, is synthetic.**

- **Interview Data Structure (for processor)**:
  Each interview should have its own subdirectory. The name of this subdirectory is the **interview name**, which should be in the format `interview_XXXX`, where `XXXX` (without underscores), is a unique identifier for the interview (e.g., `interview_food` is a folder containing interview transcript files for food theme.). While it is suggested to place these directories under `data/private_data/` for confidentiality, you may choose a different location if needed.

- **Transcript TXT Files (for processor)**:
  There are no naming requirements for the naming of the transcript files. But they must be placed directly inside the interview directory. For example:
  - `data/private_data/interview_food/transcript1.txt`
  - `data/private_data/interview_food/transcript2.txt`
  - etc.  

- **Guidelines CSV File (for processor)**:
  A CSV file named `interview_xxx_guidelines.csv` containing the guideline questions. There should be a column named `guide_text` with the guideline questions. For example:
  - `interview_food_guidelines.csv` contains the guideline questions for the interviews of food theme.

- **Reference Answer (for evaluation)**: A CSV file named `response_xxx.csv` containing the reference answers for the guideline questions. The first column should be `respondent_id`, and the remaining columns should be the reference answers to the corresponding guideline questions. For example:
  - `response_food.csv` contains the reference answers for the food theme.



## Data Format and Sample Data Output for D2D Pipeline

### Data Input
The D2D pipeline (processor part) processes two types of input files to extract and structure responses from unstructured interview transcripts based on provided guidelines.

### 1. Guidelines
- **Description**: A structured file listing the questions or prompts to guide the interview and extraction process.
- **Format**: Comma-separated values (`.csv`)
- **Structure**:
  - Single column named `guide_text` with each row containing a question or prompt.
  - Questions align with those asked in transcripts for matching purposes.
- **Example**:
  - **File**: Extract from [interview_food_sample_guidelines.csv](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food_guidelines.csv)
    > `guide_text`  
    What’s a dish that reminds you of your childhood?  
    Can you describe a meal that has a special meaning for you?  
    ...

### 2. Transcripts
- **Description**: Raw text files containing conversational interview data, with alternating lines or labeled segments for interviewers and interviewees.
- **Format**: Plain text (`.txt`)
- **Structure**:
  - Each file represents one interview.
  - Content includes dialogue, with questions from interviewers and responses from interviewees.
- **Example**:
  - **File**: Extract from [001.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food/001.txt)
    > Interviewer: Let’s talk food. What’s a dish that reminds you of your childhood?  
    > Interviewee: Definitely my grandma’s chicken and rice. She used to make it every Sunday, and the smell would just take over the whole house. It was simple—nothing fancy—but it was filled with love.  
    > Interviewer: Can you describe a meal that has a special meaning for you?  
    > Interviewee: Yeah, actually. My 18th birthday dinner. My parents surprised me by cooking all my favorite dishes—pad thai, roasted veggies, and this chocolate lava cake I was obsessed with. I remember feeling really seen, you know?
    > ...
  - **File**: Extract from [002.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food/002.txt)
    > Interviewer: Alright, diving into food and memories—what dish instantly brings your childhood back?  
    > Interviewee: Oh man, my mom’s arroz con leche. She’d make it every time I was sick, or honestly, just when I needed cheering up. The cinnamon smell still makes me emotional sometimes.  
    > Interviewer: Can you describe a meal that holds special meaning for you?  
    > Interviewee: Our Christmas Eve dinner. It’s this big spread—tamales, roasted pork, rice, beans. It’s loud and chaotic and full of stories. It’s more than food—it’s our whole culture on a table.
    > ...



### Sample Data Output
The D2D pipeline produces structured output by matching interviewee responses to guideline questions, consolidating results for analysis.
- **Format**: Comma-separated values (`.csv`)
- **Structure**:
  - Columns:
    - `Interview File`: Identifier of the source transcript file (e.g., `001`, `002`).
    - Additional columns named after guideline questions (e.g., "What’s a dish that reminds you of your childhood?").
  - Each row corresponds to one interview, with cells containing the extracted response text.
  - Responses are concise, summarizing key points from the transcript.  
- **Example**:  

| **Interview File** | **What’s a dish that reminds you of your childhood?** | **Can you describe a meal that has a special meaning for you?**                          | ... |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ | --- |
| 001            | Grandma’s chicken and rice                       | 18th birthday dinner with favorite dishes cooked by parents.                        |...|
| 002            | Mom’s arroz con leche                            | Christmas Eve dinner with tamales, roasted pork, rice, beans; loud, chaotic, full of stories. |...|
|...|...|...|...|



## How It Works (Processor part)

The processor follows these steps:
1. **Segmentation**: Divides the transcript into question-response pairs.
2. **Summarization**: Summarizes the questions in the transcript and guideline questions.
3. **Embedding**: Uses a SentenceTransformer model to embed summarized questions in the transcript and guideline questions.
4. **Matching**: Matches segments to guideline questions via cosine similarity.
5. **Summarization**: Summarizes matched segments using an LLM.
6. **Output**: Generates a CSV with summaries, plus JSON metadata and a log file.



## Usage

To run the processor on the synthetic data, use the following command after setting up your environment and data:

```bash
python examples/api_test/processor_test.py
```

**Note: To test different scenarios, navigate to `processor_test.py` and uncomment the relevant function you want to run in the main function. To ensure clarity, please run one function at a time. For more details, refer to the comments for each function in `processor_test.py`.**


## Output Storage
The output CSV file will be generated and stored in the `results/` directory. Due to confidentiality, this file should not be pushed to the repository. The `.gitignore` file is already configured to exclude the `results/` directory, so you don’t need to worry about accidentally committing sensitive output files.


## Evaluator

## Data Format and Sample Data Output

### Data Input
The D2D pipeline (evaluator part) evaluates the performance of the processor and scores the pipeline result with 5 metrics (faithfulness, correctness, precision, recall, and relevance) and a weighted join score. 
The evaluator takes 2 outputs of the processor and a reference answer as inputs.

### 1. CSV Output from Processor
- **Description**: A CSV file of the structured output of the D2D pipeline for the given dataset. Each row typically corresponds to a single transcript,
  with columns containing summarized responses aligned with the guideline questions from the interview guidelines.
- **Format**: Comma-separated values (`.csv`)
- **Structure**:
  - Columns:
    - `Interview File`: Identifier of the source transcript file (e.g., `001`, `002`).
    - Additional columns named after guideline questions (e.g., "What’s a dish that reminds you of your childhood?").
  - Each row corresponds to one interview, with cells containing the extracted response text.
  - Responses are concise, summarizing key points from the transcript.  
- **Example**:
  - **File**: `D2D_survey_food_responses_*.csv (located in the results/ directory by default)`
  
| **Interview File** | **What’s a dish that reminds you of your childhood?** | **Can you describe a meal that has a special meaning for you?**                          |...|
|----------------|--------------------------------------------------|------------------------------------------------------------------------------------|---|
| 001            | Grandma’s chicken and rice                       | 18th birthday dinner with favorite dishes cooked by parents.                        |...|
| 002            | Mom’s arroz con leche                            | Christmas Eve dinner with tamales, roasted pork, rice, beans; loud, chaotic, full of stories. |...|
|...|...|...|...|

### 2. Log (TXT) Output from Processor
- **Description**: The log file records the processing steps applied to the given interview dataset within the D2D pipeline.
  It captures details such as segmentation, embedding, matching, and summarization, along with any informational messages, warnings,
  or errors encountered during execution.
- **Format**: Plain text (`.txt`)
- **Structure**:
  - Each chunk of text marked with `===Start===` and `===End===` includes one analyzed guideline question from a single interview file.
  - Each chunk consists of the file name, guideline question, and relevant chunks of questions and answers that may contain the response. 
- **Example**:
  - **File**: `D2D_survey_food_generator_log_*.txt (located in the results/ directory by default)`
  > ===Start===  
  > Processing file: 002  
  > Processing guide question: How do food and family traditions connect for you?  
  > Relevant Interviewee Responses:  
  > Interviewer: What about food and family traditions—how do they connect?  
  > Interviewee: They’re basically the same thing in my family. Recipes are sacred. Like, if you try to tweak my aunt’s flan recipe, you might start a family feud. [laughs]  
  > Interviewer: Any food tied to a place or person for you?  
  > Interviewee: Yeah—empanadas always remind me of my grandma in Buenos Aires. She’d let me help fold the dough, and I’d sneak bits of the filling when she wasn’t looking.  
  > Interviewer: Favorite dish from another culture?  
  > Interviewee: Japanese ramen. The broth, the noodles, the toppings—it’s like a bowl of magic. I tried making it once. Total disaster. [laughs]  
  > Interviewer: Alright, diving into food and memories—what dish instantly brings your childhood back?  
  > Interviewee: Oh man, my mom’s arroz con leche. She’d make it every time I was sick, or honestly, just when I needed cheering up. The cinnamon smell still makes me emotional sometimes.  
  > Interviewer: What’s the first thing you learned to cook?  
  > Interviewee: French toast! I was like nine, and I made it for my dad on Father’s Day. I used way too much cinnamon, but he ate it like it was gourmet. I’ll never forget that.  
  > ===End===
  
### 3. CSV Input from Reference
- **Description**: A CSV file containing the reference ground truth. This file serves as the correct output to evaluate the performance of the model output.
- **Format**: Comma-separated values (`.csv`)
- **Structure**:
  - Columns:
    - `respondent_id`: Identifier of the source transcript file (e.g., `001`, `002`).
    - Additional columns named after guideline questions (e.g., "What’s a dish that reminds you of your childhood?").
  - Each row corresponds to one interview, with cells containing the extracted response text.
- **Example**:
  - **File**: Extract from [responses_food.csv](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/responses_food.csv)
  
| **respondent_id** | **What’s a dish that reminds you of your childhood?** | **Can you describe a meal that has a special meaning for you?**                          |...|
|----------------|--------------------------------------------------|------------------------------------------------------------------------------------|---|
| 001            | Grandma’s chicken and rice                       | 18th birthday dinner when family made all my favorite foods.                        |...|
| 002            | Mom’s arroz con leche                            | Christmas Eve dinner with tamales, pork, beans, family stories. |...|
|...|...|...|...|

## How It Works (Evaluator part)

The evaluator works with these steps:
1. **Merging**: Convert all files to long format (1 row per Q&A) merge on `respondent_id` and `question`.
2. **Prompting**: Build GPT prompts per metric using:Question, Model answer, Reference answer (if any), and Retrieved context.
3. **GPT Scoring**: Send prompts to GPT to get scores and explanations on faithfulness, correctness, relevance, precision, and recall.
4. **Output**: Save scored results to a CSV file.
5. **Post-Processing**: Highlights any rows with any score less than a specified threshold, and computes a weighted joint score.
6. **End Output**: 4 csv files of retrieved contexts, scores and feedback, highlighted low scores, and joint metric scores. 

## Usage

To run the evaluator on the synthetic data, use the following command after running the processor and obtained output from processor:

```bash
python examples/api_test/evaluator_test.py
```

## Data Output

After finishing the evaluation process, the evaluator generates 4 output `csv`s. In the example, the `csv`s are stored under the `eval_results/` directory. For the synthetic interview dataset `interview_food`, these following files are outputted:

- **`eval_output_post_joint_metric.csv`**
  - **Description**: An aggregated summary showing the average score per metric for each respondent, along with a joint score calculated using user-defined metric weights that sum to 1.0. The joint score reflects an overall performance index.

- **`eval_output_post_highlighted.csv`**
  - **Description**: A filtered version of the evaluation output, containing only the rows where at least one metric score falls below a specified threshold. Low scores are tagged for easy identification (e.g., `low-1.0`).

- **`eval_output.csv`**
  - **Description**: This is the core evaluation output generated after running the GPT-based scoring pipeline. Each row represents a (`respondent_id`, `question`) pair and includes the model-generated answer, reference answer (if available), retrieved context, and evaluation results for each enabled metric.

- **`retrieved_contexts.csv`**
  - **Description**: A file that logs the retrieved chunks or evidence segments for each (`respondent_id`, `question`) pair. These context chunks are used to support the model's generation and are referenced during evaluations for metrics like faithfulness, precision, and recall.

## Detailed Documentation
For more detailed Documentation, please navigate to [here](https://github.com/avalanche-strategy/D2D/blob/main/docs/example.ipynb)

## Dependencies

- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)
- `jupyterlab` (version 4.0.0 or higher)
- `nb_conda_kernels` (version 2.3.1 or higher)
- Python and packages listed in [`environment.yml`](environment.yml)
- [Docker](https://www.docker.com/)

### Adding a new dependency

1. Add the dependency to the `environment.yml` file on a new branch.

2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.

2. Re-build the Docker image locally to ensure it builds and runs properly.

3. Push the changes to GitHub. A new Docker
   image will be built and pushed to Docker Hub automatically.

5. Send a pull request to merge the changes into the `main` branch. 

## Contributing
Interested in contributing? Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute. Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## Authors

- Sienko Ikhabi 
- Dominic Lam
- Yun Zhou
- Wangkai Zhu

## License
This project is licensed under the Apache License, Version 2.0 - see LICENSE for details.

## Project Status
D2D is actively developed. We welcome feedback and feature requests via GitHub issues.
