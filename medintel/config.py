import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM Configuration
LLM_MODEL = os.getenv("MEDINTEL_LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("MEDINTEL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("MEDINTEL_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

# Retrieval Parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
TOP_K_DENSE = 6
TOP_K_SPARSE = 6
RRF_K_CONSTANT = 60
TOP_K_RERANKED = 3

# Chroma DB Path
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
