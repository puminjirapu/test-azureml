from dotenv import load_dotenv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # MLOps directory
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Check if environment variables are loaded
subscription_id = os.getenv("TARGET_COL")
print(f"Subscription ID: {subscription_id}")