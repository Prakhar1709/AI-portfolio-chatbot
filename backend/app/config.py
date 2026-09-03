import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# Load .env file if available
load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "Prakhar AI Portfolio Chatbot Engine"
    API_V1_PREFIX: str = "/api"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    KNOWLEDGE_DIR: Path = KNOWLEDGE_DIR
    CHROMA_PERSIST_DIR: Path = CHROMA_PERSIST_DIR
    CHROMA_COLLECTION_NAME: str = "prakhar_portfolio_knowledge"
    
    # API Keys & LLM Providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Default Provider Selection
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")  # "gemini", "openai", "groq", "offline"
    DEFAULT_EMBEDDING_PROVIDER: str = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "gemini")  # "gemini", "openai", "local"
    
    # Retrieval Tuning
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    
    # Persona Details
    PERSONA_NAME: str = "Prakhar"
    PERSONA_TITLE: str = "AI/ML Developer & GenAI Engineer"
    LOCATION: str = "Faridabad, India"
    COLLEGE: str = "IIITDM Jabalpur (Batch of 2027)"

settings = Settings()
