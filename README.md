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
3. Install the package:
```bash
pip install .
```

## Usage
```python
# To be filled
```

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
Interested in contributing? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute. Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## Authors

- Sienko Ikhabi 
- Dominic Lam
- Yun Zhou
- Wangkai Zhu

## License
This project is licensed under the Apache License, Version 2.0 - see LICENSE for details.
