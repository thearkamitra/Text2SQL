from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv, dotenv_values
import os

# LangChain imports
# Note: For Ollama, install the new package: pip install langchain-ollama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.llms.base import LLM
from langchain_groq import ChatGroq

class LLMProvider(Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"


class BaseLLMAgent(ABC):
    """
    Abstract base class for LLM generators.
    Provides a common interface for creating different LangChain LLM instances.
    """

    def __init__(self, model_name: str, env_path_name: Optional[str] = None, **kwargs):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the model to use
            env_path_name: Path to environment file containing API keys
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.env_path_name = env_path_name
        if self.env_path_name is None:
            self.env_path_name = os.path.join(os.path.dirname(__file__), ".env")            
        self.config = kwargs
        if not load_dotenv(self.env_path_name):
            raise FileNotFoundError(f"Environment file not found at {self.env_path_name}")
        else:
            self.env_keys = dotenv_values(self.env_path_name)

    @abstractmethod
    def create_llm(self) -> LLM:
        """
        Create and return a LangChain LLM instance.
        
        Returns:
            LangChain LLM instance ready for use
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        pass


class OpenAIAgent(BaseLLMAgent):
    """OpenAI LLM agent implementation"""

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, **kwargs)
        self._llm = None
    
    def create_llm(self) -> LLM:
        """Create and return a LangChain ChatOpenAI instance"""
        if self._llm is None:
            api_key = self.env_keys.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment file")
            
            self._llm = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=api_key,
                temperature=0.0,
                **self.config
            )
        return self._llm
    
    def is_available(self) -> bool:
        """Check if OpenAI service is available"""
        try:
            api_key = self.env_keys.get("OPENAI_API_KEY")
            return api_key is not None and len(api_key.strip()) > 0
        except:
            return False


class GroqAgent(BaseLLMAgent):
    """Groq LLM agent implementation"""

    def __init__(self, model_name: str = "llama-3.1-8b-instant", **kwargs):
        super().__init__(model_name, **kwargs)
        self._llm = None
    
    def create_llm(self) -> LLM:
        """Create and return a LangChain ChatGroq instance"""
        if self._llm is None:
            api_key = self.env_keys.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment file")
            
            # Use ChatGroq for Groq API
            self._llm = ChatGroq(
                model_name=self.model_name,
                groq_api_key=api_key,
                temperature=0.0,
                **self.config
            )
        return self._llm
    
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        try:
            api_key = self.env_keys.get("GROQ_API_KEY")
            return api_key is not None and len(api_key.strip()) > 0
        except:
            return False


class OllamaAgent(BaseLLMAgent):
    """Ollama LLM agent implementation"""
    
    def __init__(self, model_name: str = "qwen3:0.6b", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self._llm = None
    
    def create_llm(self) -> LLM:
        """Create and return a LangChain OllamaLLM instance"""
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
                **self.config
            )
        return self._llm
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class LLMAgentFactory:
    """Factory class to create LLM agents"""
    
    @staticmethod
    def create_agent(provider: LLMProvider, model_name: Optional[str] = None, 
                    env_path_name: Optional[str] = None, **kwargs) -> BaseLLMAgent:
        """
        Create an LLM agent based on the specified provider.
        
        Args:
            provider: LLM provider type
            model_name: Name of the model (optional, uses default if not specified)
            env_path_name: Path to environment file containing API keys
            **kwargs: Additional configuration parameters
            
        Returns:
            Instance of the appropriate LLM agent
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider == LLMProvider.OPENAI:
            return OpenAIAgent(
                model_name=model_name or "gpt-4o",
                env_path_name=env_path_name,
                **kwargs
            )
        elif provider == LLMProvider.GROQ:
            return GroqAgent(
                model_name=model_name or "llama-3.1-8b-instant",
                env_path_name=env_path_name,
                **kwargs
            )
        elif provider == LLMProvider.OLLAMA:
            return OllamaAgent(
                model_name=model_name or "qwen3:0.6b",
                env_path_name=env_path_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

