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
    url="https://github.com/avalanche-strategy/D2D",  # Update as needed
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"d2d": ["config/llm_defaults.json"]},
    install_requires=[
        "torch>=1.10.0",
        "sentence-transformers>=2.2.0",
        "litellm>=1.0.0",
        "python-dotenv>=0.21.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
        "ragas>=0.1.0",
        "rapidfuzz>=3.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)