import dotenv
import os

dotenv.load_dotenv()

#Config
EMBED_MODEL  = "text-embedding-3-small"
DIMENSIONS   = 1536
LLM_MODEL    = "gemma-4-31b-it"
REWRITE_MODEL = "gemma-4-31b-it"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "anime_collection"
TOP_K        = 10
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")