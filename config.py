import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # AI Model Configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_DEFAULT_MODEL = os.getenv('OLLAMA_DEFAULT_MODEL', 'llama2')
    
    # Hugging Face Configuration
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    HUGGINGFACE_DEFAULT_MODEL = os.getenv('HUGGINGFACE_DEFAULT_MODEL', 'mistralai/Mistral-7B-Instruct-v0.1')
    
    # Deployment
    DEPLOYMENT = os.getenv('DEPLOYMENT', 'false').lower() == 'true'
