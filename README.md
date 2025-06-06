# Dialogue2Data (D2D)

## About
Dialogue2Data (D2D) is an open-source Python package that transforms unstructured interview transcripts into structured data for analysis. Using NLP and discussion guides, it automates topic matching and generates structured outputs (e.g., CSV) compatible with Fathom's survey analysis pipeline.



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
To use the OpenAI and Anthropic API, you need to set up an environment variable for your API key. Create a `.env` file in the root directory of the project with the following content:

- **Example:**  
```bash
OPENAI_API_KEY=sk-abc123XYZ789pqr456STU012vwx789YZ
ANTHROPIC_API_KEY=sk-ant-987ZYX654WVU321TSR098qwe456PLM
```

**Note: This are fictional keys.**



### Data Placement

To ensure smooth operation, please organize your data as follows:

- **Interview Data Structure (for processor)**:
  Each interview should have its own subdirectory. The name of this subdirectory is the **interview name**, which should be in the format `interview_XXXX`, where `XXXX`, WITHOUT underscore `_` in it, is a unique identifier for the interview (e.g., `interview_food` is a folder containing interview transcript files for food theme.). While it is suggested to place these directories under `data/private_data/` for confidentiality, you may choose a different location if needed.

- **Transcript TXT Files (for processor)**:
  There is no requirement for the naming of the transcript files. Just make sure all transcript files are placed directly inside the interview directory. For example:
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
  - **File**: `[interview_food_sample_guidelines.csv](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food_guidelines.csv)`
    > `guide_text`  
    > What’s a dish that reminds you of your childhood?  
    > Can you describe a meal that has a special meaning for you?
    > ...

### 2. Transcripts
- **Description**: Raw text files containing conversational interview data, with alternating lines or labeled segments for interviewers and interviewees.
- **Format**: Plain text (`.txt`)
- **Structure**:
  - Each file represents one interview.
  - Content includes dialogue, with questions from interviewers and responses from interviewees.
- **Example**:
  - **File**: `[001.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food/001.txt)`
    > Interviewer: Let’s talk food. What’s a dish that reminds you of your childhood?  
    > Interviewee: Definitely my grandma’s chicken and rice. She used to make it every Sunday, and the smell would just take over the whole house. It was simple—nothing fancy—but it was filled with love.  
    > Interviewer: Can you describe a meal that has a special meaning for you?  
    > Interviewee: Yeah, actually. My 18th birthday dinner. My parents surprised me by cooking all my favorite dishes—pad thai, roasted veggies, and this chocolate lava cake I was obsessed with. I remember feeling really seen, you know?
    > ...
  - **File**: `[002.txt](https://github.com/avalanche-strategy/D2D/blob/main/data/synthetic_data/interview_food/002.txt)`
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
| Interview File | What’s a dish that reminds you of your childhood? | Can you describe a meal that has a special meaning for you?                          |...|
|-|--|||
| 001            | Grandma’s chicken and rice                       | 18th birthday dinner with favorite dishes cooked by parents.                        |...|
| 002            | Mom’s arroz con leche                            | Christmas Eve dinner with tamales, roasted pork, rice, beans; loud, chaotic, full of stories. |...|
|...|...|...|...|



## How It Works (Processor part)

The processor follows these steps:
1. **Segmentation**: Divides the transcript into question-response pairs.
2. **Summarization**: Summarize the questions in the transcript and guideline questions.
3. **Embedding**: Uses a SentenceTransformer model to embed summarized questions in the transcript and guideline questions.
4. **Matching**: Matches segments to guideline questions via cosine similarity.
5. **Summarization**: Summarizes matched segments using an LLM.
6. **Output**: Generates a CSV with summaries, plus JSON metadata and a log file.



## Usage

To run the processor on the synthetic data, use the following command after setting up your environment and data:

```bash
python examples/api_test/processor_test.py
```

**Note: To test different scenarios, navigate to `processor_test.py` and uncomment the relevant functions in the main function.**


## Output Storage
The output CSV file will be generated and stored in the `results/` directory. Due to confidentiality, this file should not be pushed to the repository. The `.gitignore` file is already configured to exclude the `results/` directory, so you don’t need to worry about accidentally committing sensitive output files.


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
