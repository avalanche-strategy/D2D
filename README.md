# Dialogue2Data (D2D)

## About
Dialogue2Data (D2D) is an open-source Python package that transforms unstructured interview transcripts into structured data for analysis. Using NLP and discussion guides, it automates topic matching and generates structured outputs (e.g., CSV) compatible with Fathom's survey analysis pipeline.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/fathomthat/d2d.git
cd D2D
```
2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate d2d
```

## Environment Configuration 
To use the OpenAI and Anthropic API, you need to set up an environment variable for your API key. Create a `.env` file in the root directory of the project with the following content:

```bash
OPENAI_API_KEY=[Please replace this with your OPENAI API key]
ANTHROPIC_API_KEY=[Please replace this with your ANTHROPIC API key]
```

Make sure to replace `[Please replace this with the OPENAI API key]` with your actual OpenAI API key. This file should not be committed to the repository, as it contains sensitive information.

### Data Placement

To ensure smooth operation of the processor, please organize your data as follows:

- **Interview Data Structure**:  
  Each interview should have its own subdirectory. The name of this subdirectory is the **interview name**, which MUST be in the format `interview_XXXX`, where `XXXX` is a unique identifier for the interview (e.g., `interview_1090`). While it is suggested to place these directories under `data/private_data/` for confidentiality, you may choose a different location if needed.

- **Transcript Files**:  
  Place all transcript files directly inside the interview directory. For example:  
  - `data/private_data/interview_1090/transcript.txt`

- **Guidelines CSV File**:  
  The guidelines CSV file must be placed in the same level of the interview directory and should be named consistently with the interview name. For example:  
  - `data/private_data/interview_1090_guidelines.csv`

- **Synthetic Data**:  
  For demonstration purposes, synthetic data is provided in `data/synthetic_data/interview_food`. This can be used to test the processor without needing private data.

## Usage

To run the processor on your data, use the following command after setting up your environment and data:

```python
python examples/api_test/processor_test.py
```

This script processes the transcript and guidelines in the specified interview directory and generates a structured CSV output. More detailed usage instructions will be added as the project develops.

## Output Storage
The output CSV file will be generated and stored in the `results/` directory. Due to confidentiality, this file should not be pushed to the repository. The `.gitignore` file is already configured to exclude the `results/` directory, so you don’t need to worry about accidentally committing sensitive output files.


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
