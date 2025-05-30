import os
from dotenv import load_dotenv
# Manually load .env file from the current working directory
dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    with open(dotenv_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
else:
    print(f".env file not found at {dotenv_path}")

# Make sure it loads the .env file if it's not from the current working directory
load_dotenv()

from .processor import D2DProcessor
from .evaluator import D2DEvaluator

__all__ = ["D2DProcessor", "D2DEvaluator"]