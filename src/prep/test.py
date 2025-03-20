from dotenv import load_dotenv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # MLOps directory
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Check if environment variables are loaded
TARGET_COL = os.getenv("TARGET_COL")
print(TARGET_COL)

