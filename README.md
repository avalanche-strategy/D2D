# Dialogue2Data (D2D)

Dialogue2Data (D2D) is an open-source Python package that transforms unstructured interview transcripts into structured data for analysis. It consists of two major components: 

- The **Processor**, which leverages natural language processing (NLP), large language models (LLMs), and sentence embeddings to automate topic matching, response extraction, and summarization based on discussion guides (guideline questions). The Processor generates structured outputs (e.g., CSV, JSON) that can then be analyzed using any data analysis tools, like `pandas`.
- The **Evaluator**, which assesses output quality using metrics like faithfulness, correctness, precision, recall, and relevance.

D2D is ideal for researchers and analysts processing qualitative interview data.


## Installation

**Note:** These instructions assume you have `git` installed on your system. If not, please install it from [https://git-scm.com/](https://git-scm.com/).

To install this package locally with `pip`, follow these steps:

### **1. Clone the repository:**

```bash
git clone https://github.com/avalanche-strategy/D2D.git
```

---


### **2. Navigate to the top-level directory of the new repository:**

```bash
cd D2D
```

---

### **3. Set up your environment**

**Create a new Conda environment with Python 3.10:**

```bash
conda create -n d2d-310 python=3.10
conda activate d2d-310
python --version 
```

Make sure the output is **Python 3.10.x** or higher.

**Note**: We use Python 3.10 here to test compatibility, but you can use any Python version 3.10 or higher.  

---

### **4. Install the package**

After you have Python 3.10 or above set up, from the project root directory (where `setup.py` is located), run:

```bash
pip install .
```

### Clean Up / Uninstall
After you have finished testing or using the package, you may want to: 
1. Uninstall `d2d`:
```bash
pip uninstall d2d
```
2. Deactivate the virtual environment:
```bash
conda deactivate
```


## Environment Configuration
To use the OpenAI and Anthropic APIs, you need to set up an environment variable for your API key. Create a `.env` file in the root directory of the project with the following content:

- **Example:**  
```bash
OPENAI_API_KEY=sk-abc123XYZ789pqr456STU012vwx789YZ
ANTHROPIC_API_KEY=sk-ant-987ZYX654WVU321TSR098qwe456PLM
```

**Note: These are fictional keys. To ensure smooth operation, please use your own API keys.** 

You do not need to set up keys for both APIs. If you will only use one, set up the key for just that API.


### Data Placement

To ensure smooth operation, please organize your data as follows:

**Note: All sample data in this repository is synthetic and safe to use for demos and tests.**

- **Interview Data Structure (for processor)**:

  As input, the processor requires the guidelines CSV file and the name of a directory that contains the transcript TXT files.
  
  - **Transcript TXT Files**:
    There are no requirements for the naming either the transcript folder nor the transcript files within the folder. The only expectation is that the TXT files containing individual interview transcript text must have the `*.txt` extension and be placed directly inside the interview transcripts directory (not nested in child sub-directories). For example:
    - `data/private_data/interview_food/transcript1.txt`
    - `data/private_data/interview_food/transcript2.txt`
    - etc. 
    You will then need to pass the name of the directory (`data/private_data/interview_food`) in the example above, to the processor. You may pass a relative path, as in the example, or an absolute path (e.g. `/home/user/data/private_data/interview_food`).

  - **Guidelines CSV File**:
    A CSV file containing the guideline questions. The CSV file must contain a column named `guide_text` with the guideline questions. There are no naming restrictions by the processor. However, for your own organization and keeping track of guidelines and transcript folder pairing (if you manage several interviews), it might be easy to align the naming. For example:
    - `interview_food_guidelines.csv` contains the guideline questions for the interviews of food theme.

  - **Output Folder**:
    The folder (e.g. `/home/user/data/results/`) where the processor output files will be created. The folder must already exist, but the output files described below will be created. To allow multiple runs, the output filenames are generated based on the name of the interview and include the timestamp when they were generated (e.g. `D2D_survey_food_responses_2025-06-21_15-21.csv`).

- **Reference Answer (for evaluation, optional)**: If you will plan to evaluate the output of the processor, you will need to prepare a CSV file named `response_xxx.csv`. This file will contain the reference answers for the guideline questions. The first column should be `respondent_id`, corresponding to the interview filename (e.g. `transcript1` for the responses you expect from `transcript1.txt`), and the remaining columns should be the reference answers to the corresponding guideline questions. For example:
  - `response_food.csv` contains the reference answers for the food theme.



## Data Format and Sample Data Output for D2D Pipeline

### Data Input
The D2D pipeline (processor part) processes two types of input files to extract and structure responses from unstructured interview transcripts based on provided guidelines.

### 1. Guidelines
- **Description**: A structured file listing the questions or prompts to guide the interview and extraction process.
- **Format**: Comma-separated values (`.csv`)
- **Structure**:
  - Single column named `guide_text` with each row containing a question or prompt.
  - The questions should semantically align with those asked in transcripts for matching purposes.
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
  - Each file represents one two-person interview (Interviewer and Interviewee).
  - Content includes dialogue, with questions from interviewers and responses from interviewees.
- **Example**:
  - **File**: Extract from [001.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/transcripts_food/001.txt)
    > Interviewer: Let’s talk food. What’s a dish that reminds you of your childhood?  
    > Interviewee: Definitely my grandma’s chicken and rice. She used to make it every Sunday, and the smell would just take over the whole house. It was simple—nothing fancy—but it was filled with love.  
    > Interviewer: Can you describe a meal that has a special meaning for you?  
    > Interviewee: Yeah, actually. My 18th birthday dinner. My parents surprised me by cooking all my favorite dishes—pad thai, roasted veggies, and this chocolate lava cake I was obsessed with. I remember feeling really seen, you know?
    > ...
  - **File**: Extract from [002.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/transcripts_food/002.txt)
    > Interviewer: Alright, diving into food and memories—what dish instantly brings your childhood back?  
    > Interviewee: Oh man, my mom’s arroz con leche. She’d make it every time I was sick, or honestly, just when I needed cheering up. The cinnamon smell still makes me emotional sometimes.  
    > Interviewer: Can you describe a meal that holds special meaning for you?  
    > Interviewee: Our Christmas Eve dinner. It’s this big spread—tamales, roasted pork, rice, beans. It’s loud and chaotic and full of stories. It’s more than food—it’s our whole culture on a table.
    > ...



### Sample Data Output

#### CSV (Responses)

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

#### JSON (Responses and References)

In addition to the CSV output, the D2D pipeline produces structured JSON output that provides more details about each of the final response in the CSV file.
- **Format**: JavaScript Object Notation (`.json`)
- **Structure**:
  - An array of interview output objects, analogous to the rows in the CSV file. Each object corresponds to an `Interview File` and has the following attributes:
    - `interview`: An identifier of the source transcript file (e.g., `001`, `002`).
    - `transcript`: The full path of the filename referenced for the responses (e.g., `/home/user/data/private_data/interview_food/transcript1.txt`).
    - `responses`: An array of response output objects, analogous to the columns in the CSV file. Each object corresponds to a `Guideline question` and will also contain elements that are extracted from the corresponding `Interview File` in response to that `Guideline question`. Each object has the following attributes:
      - `guide_question`: The guideline question for which this response refers to.
      - `relevant_lines`: An array with tuples of line numbers from the transcript TXT file that semantically match the guideline question. These lines would have been identified by the algorithm to be most semantically similar to the guideline question. For each line number tuple, the first pair is the "Interviewer" line number and the second pair is the "Interviewee" line number. For example, if `relevant_lines` contains the value `[[4, 5], [10, 12]]`, there were two segments of the interview that matched the guideline question. First, the dialogue on line 4(Interviewer)+line 5(Interviewee) and then line 10(Interviewer)+line 12(Interviewee).
      - `extracted_phrase`: A phrase extracted from the transcript that can directly answer the `guide_question` (or the text, `[No relevant response]`, if none is found).
      - `response`: This is the final concise response from the `processor`. Note that this is the same value that is also output to the CSV file.
      - `match_type`: Text enumeration indicating how you can find `extracted_phrase` in the transcript.
        - "EXACT": The text can be found (as a whole and without modification), using a simple search e.g. using Python's `str.contains`. For instance, if the phrase that was extracted were _"mom’s arroz con leche"_, you can search the corresponding trascript text file and find this text.
        - "PARTIAL": To find the text, you need to split the `extracted_phrase` and search for each sub-segments of the phrase. For instance, if the extracted text were _"Scrambled eggs that dad taught me when I was seven"_ you can find the referenced text through two partial searches for _"Scrambled eggs"_ and _"my dad taught me when I was seven"_. The word "that" was included by the LLM to join the two phrases but is not in the original text, and it is possible that the original text includes other non-relevant words between the two phrases (in his example "...eggs! _\[laughs\]_ My dad...")
        - "SEMANTIC": No exact or partial text matches exist. However, the extracted phrase has the same meaning as a given line in the transcript. For instance, a response of _"The interviewee reported that work is going very well."_ when the transcript has the line _"Interviewee: Work is really fantastic at the moment!"_
        - "NONE": If no references were found
      - `extracted_line_references`: An array of all the line numbers from which the `extracted_phrase` was obtained (e.g., just one line - `[7]` or two different lines `[9, 11]`). These will only be based on the Interviewee responses
      - `extracted_character_index`: An array with a triple of character index references in the text. Each triple contains:
        - `line`: The line number (e.g., `17`, `33`)
        - `start`: The position/character within the line text that marks the start of the match. For instance `34` means that the text matches from character number 34 on the specified `line` number. If the match is "SEMANTIC" this will always be `-1`.
        - `end`: The position/character within the line text that marks the end of the match. For instance `61` means that the text matches from character number 34 (`start`) to 61 (`end`) on the specified `line` number. If the match is "SEMANTIC" this will always be `-1`.
        Note that if several portions/segments of the text are found on one line, the line number will be repeated in different `extracted_character_index` triples, but reported just once under `extracted_line_references`.
     
_Line numbers and character indexes are numbered from 1. Character Index will be counted **after** the marker "Interviewee: "._

- **Example**:

```json
[
  {
    "interview": "001",
    "transcript": "/home/userdata/synthetic_data/transcripts_food/001.txt",
    "responses": [
      {
        "guide_question": "What’s a dish that reminds you of your childhood?",
        "relevant_lines": [[1, 3], [13, 15], [21, 23]],
        "extracted_phrase": "My grandma’s chicken and rice",
        "response": "Grandma’s chicken and rice",
        "match_type": "EXACT",
        "extracted_line_references": [3],
        "extracted_character_index": [
          {
            "line": 3,
            "start": 11,
            "end": 40
          }
        ]
      },
      {
        "guide_question": "Can you describe a meal that has a special meaning for you?",
        "relevant_lines": [[5, 7], [13, 15], [17, 19]],
        "extracted_phrase": "My 18th birthday dinner. My parents surprised me by cooking all my favorite dishes—pad thai, roasted veggies, and this chocolate lava cake I was obsessed with",
        "response": "18th birthday dinner with favorite dishes cooked by parents.",
        "match_type": "EXACT",
        "extracted_line_references": [7],
        "extracted_character_index": [
          {
            "line": 7,
            "start": 16,
            "end": 174
          }
        ]
      }
    ]
  }
]
```

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
python examples/processor_example.py
```

**Note: To test different scenarios, navigate to [`processor_examples.py`](https://github.com/avalanche-strategy/D2D/blob/dom/final_report_cut_words/src/d2d/processor.py) and uncomment the relevant function you want to run in the main function. To ensure clarity, please run one function at a time. For more details, refer to the comments for each function in `processor_examples.py`.**

**Or, run directly from Python**

You can also use the processor in your own Python scripts as follows


```python
from d2d import D2DProcessor

processor = D2DProcessor()
processor.process_transcripts(
    transcripts_dir="path/to/transcripts",
    guidelines_path="path/to/guidelines.csv",
    interview_name="interview",
    output_dir="path/to/output"
)
```

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

To run the evaluator on the synthetic data, use the following command after running the processor and obtained the output from processor:

```bash
python examples/evaluator_example.py
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

## Running Unit Tests
Unit tests have been created for core functions of the `processor`. All the unit tests are in the folder [tests](tests). You can run the unit tests using `pytest`, specifying the directory containing the unit tests:

### 1. Install test dependencies

Before running the tests, make sure you have the required testing packages installed:

```bash
pip install -r requirements-dev.txt
```

### 2. Run the test suite

You can run all unit tests using pytest:

```bash
pytest tests/
```

**Note:**

- LLM responses can vary, so it’s normal for some tests to occasionally fail—if you see a failed test, simply try running it again.  
- Running all tests may take some time, especially those involving LLM calls.
- Make sure you are in the project’s root directory and your virtual environment is activated (if you’re using one). 
- You can modify the unit tests to include additional scenarios by adding new test functions in the existing `*.py` files or adding new tests under the folder [tests](tests).

## Detailed Documentation

- For more detailed processor and evaluator documentation, please navigate to [here](https://github.com/avalanche-strategy/D2D/blob/main/docs/user_guide.md).  
- For more detailed evaluation framework explanation, please navigate to [here](https://github.com/avalanche-strategy/D2D/blob/main/docs/evaluation_white-paper.md).  
- For more detailed tuning instructions in evaluation, please navigate to [here](https://github.com/avalanche-strategy/D2D/blob/main/docs/evaluation_tuning.md).

## Dependencies

- Python **3.10 or higher**
- All core package dependencies are installed with:
  ```bash
  pip install .
  ```
- Development and testing dependencies are listed in [requirements-dev.txt](requirements-dev.txt)

### Adding a new dependency

1.	Add the new package to requirements-dev.txt on a new branch.

Install dev dependencies in your (activated) virtual environment:

```bash
pip install -r requirements-dev.txt
```

3. Re-run all unit tests to ensure the pipeline runs properly.

```bash
pytest tests
```

4. Push the changes to GitHub and create a pull request to merge the changes into the `main` branch. 

## Warning Suppression

This project integrates with libraries (such as LiteLLM and Pydantic v2+) that may emit noisy `Pydantic serializer warnings` regarding unexpected serialization fields when handling LLM responses.

To keep logs clean and focus on actionable errors, we programmatically suppress these specific warnings at package import time (see `d2d/__init__.py`). This does not affect core processing or outputs.

If you modify LLM integration code, serialization logic, or upgrade Pydantic/LiteLLM, please review this suppression logic and consider removing or updating it if no longer needed.

## Contributing
Interested in contributing? Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute. Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## Authors

- Sienko Ikhabi 
- Dominic Lam
- Yun Zhou
- Wangkai Zhu

## License
This project is licensed under the Apache License, Version 2.0 - see LICENSE for details.

