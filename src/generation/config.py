import os

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in the environment."
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

TEMPERATURE = 0.0

MAX_TOKENS = 800


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

TOP_K = 5

MIN_HYBRID_SCORE = 0.15