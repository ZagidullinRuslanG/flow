import os
import requests
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"ollama_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("ollama_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        n_ctx: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Base URL for Ollama API (defaults to OLLAMA_BASE_URL env var or 'http://localhost:11434')
            model: Model name to use (defaults to OLLAMA_MODEL env var or 'llama2')
            n_ctx: Context window size (defaults to OLLAMA_N_CTX env var or 4096)
            temperature: Sampling temperature (0.0 to 1.0, defaults to OLLAMA_TEMPERATURE env var or 0.7)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self.n_ctx = n_ctx or int(os.getenv("OLLAMA_N_CTX", "4096"))
        self.temperature = temperature or float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

        logger.info(f"base_url: {self.base_url}")
        logger.info(f"model: {self.model}")
        logger.info(f"n_ctx: {self.n_ctx}")
        logger.info(f"temperature: {self.temperature}")

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            temperature: Override default temperature for this request
            
        Returns:
            Generated text
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_ctx": self.n_ctx,
                    "temperature": temperature if temperature is not None else self.temperature
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def list_models(self) -> list:
        """
        List available models
        
        Returns:
            List of available models
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json()["models"]
    
    def pull_model(self, model_name: str) -> None:
        """
        Pull a model from Ollama hub
        
        Args:
            model_name: Name of the model to pull
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name}
        )
        response.raise_for_status() 