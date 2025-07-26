"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # Supabase configuration
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # API Keys for LLM Services
    GROQ_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    TOGETHER_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    
    # LLM Model configuration
    LLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.1"
    PARSER_MODEL_NAME: str = "llama3-8b-8192"
    LLM_FEEDBACK_MODEL_NAME: str = "mistralai/mistral-7b-instruct:free"
    
    # Embedding model configuration
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # API configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: list = ["application/pdf"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings() 