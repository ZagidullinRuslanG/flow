import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"openrouter_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("openrouter_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Base URL for OpenRouter API (defaults to OPENROUTER_BASE_URL env var or 'https://openrouter.ai/api/v1')
            model: Model name to use (defaults to OPENROUTER_MODEL env var or 'anthropic/claude-3-opus-20240229')
            max_tokens: Maximum number of tokens to generate (defaults to OPENROUTER_MAX_TOKENS env var or 4096)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
            
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-opus-20240229")
        self.max_tokens = max_tokens or int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))

        logger.info(f"base_url: {self.base_url}")
        logger.info(f"model: {self.model}")
        logger.info(f"max_tokens: {self.max_tokens}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        stream: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using OpenRouter API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response
            additional_params: Additional parameters to pass to the API
            
        Returns:
            Generated text
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "My LLM App")
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

        if additional_params:
            payload.update(additional_params)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        if stream:
            return self._handle_stream(response)
        else:
            return response.json()["choices"][0]["message"]["content"]

    def _handle_stream(self, response):
        """
        Handle streaming response from OpenRouter API
        
        Args:
            response: Response object from requests
            
        Returns:
            Complete generated text
        """
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = requests.json.loads(data)
                        if chunk['choices'][0]['finish_reason'] is not None:
                            break
                        content = chunk['choices'][0]['delta'].get('content', '')
                        full_text += content
                    except requests.json.JSONDecodeError:
                        continue
        return full_text

    def list_models(self) -> list:
        """
        List available models
        
        Returns:
            List of available models
            
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "My LLM App")
        }
        
        response = requests.get(
            f"{self.base_url}/models",
            headers=headers
        )
        response.raise_for_status()
        return response.json()["data"] 