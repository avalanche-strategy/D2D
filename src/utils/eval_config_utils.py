import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("api_key"))

# Metric toggle dictionary
ACTIVE_METRICS = {
    "faithfulness": True,
    "precision": True,
    "recall": True,
    "relevance": True,
    "correctness": True
}
