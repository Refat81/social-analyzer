import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', '')
    
    # Hugging Face Configuration (Primary for deployment)
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    HUGGINGFACE_DEFAULT_MODEL = os.getenv('HUGGINGFACE_DEFAULT_MODEL', 'mistralai/Mistral-7B-Instruct-v0.1')
    
    # Ollama Configuration (Fallback for local development)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_DEFAULT_MODEL = os.getenv('OLLAMA_DEFAULT_MODEL', 'llama2')
    
    # Deployment
    DEPLOYMENT = os.getenv('DEPLOYMENT', 'false').lower() == 'true'
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if cls.DEPLOYMENT and not cls.HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY is required for deployment")
        return True
