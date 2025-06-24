from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="d2d",
    version="0.2.8",
    author="Sienko Ikhabi, Dominic Lam, Yun Zhou, Wangkai Zhu",
    author_email="your.email@example.com",
    description="Dialogue2Data: Transform interview transcripts into structured data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avalanche-strategy/D2D",  # Replace with your repo URL
    package_dir={"": "src"},
    package_data={'d2d': ['config/llm_defaults.json']},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "torch>=1.10.0",  # For tensor operations with sentence transformers
        "sentence-transformers>=2.2.0",  # For embedding models
        "litellm>=1.0.0",  # For LLM interactions
        "python-dotenv>=0.21.0",  # For environment variable management
        "pandas>=1.5.0",  # For data manipulation
        "numpy>=1.23.0",  # For numerical operations
        "tqdm>=4.65.0",  # For progress bars
        "openai>=1.0.0",  # For OpenAI API client
        "ragas>=0.1.0",  # For evaluation (run_ragas_evaluation)
        "rapidfuzz>=3.9.0",  # For fuzzy string matching
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",

)