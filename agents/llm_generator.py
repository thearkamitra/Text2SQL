from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv, dotenv_values
import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.llms import OllamaLLM as Ollama
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
    
    def __init__(self, model_name: str = "llama3-8b-8192", **kwargs):
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
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self._llm = None
    
    def create_llm(self) -> LLM:
        """Create and return a LangChain Ollama instance"""
        if self._llm is None:
            self._llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
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
                model_name=model_name or "llama3-8b-8192",
                env_path_name=env_path_name,
                **kwargs
            )
        elif provider == LLMProvider.OLLAMA:
            return OllamaAgent(
                model_name=model_name or "llama2",
                env_path_name=env_path_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Usage example:
if __name__ == "__main__":
    # Example usage
    try:
        # Create OpenAI agent
        openai_agent = LLMAgentFactory.create_agent(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="your-openai-api-key"
        )

        # Get the LangChain LLM instance
        openai_llm = openai_agent.create_llm()
        print(f"Created OpenAI LLM: {type(openai_llm)}")
        
        # Create Groq agent
        groq_agent = LLMAgentFactory.create_agent(
            provider=LLMProvider.GROQ,
            model_name="llama3-8b-8192"
        )
        
        # Get the LangChain LLM instance
        groq_llm = groq_agent.create_llm()
        print(f"Created Groq LLM: {type(groq_llm)}")
        
        # Create Ollama agent (no API key needed for local instance)
        ollama_agent = LLMAgentFactory.create_agent(
            provider=LLMProvider.OLLAMA,
            model_name="llama2"
        )
        
        # Get the LangChain LLM instance
        ollama_llm = ollama_agent.create_llm()
        print(f"Created Ollama LLM: {type(ollama_llm)}")
        
        # Test if services are available
        print(f"OpenAI available: {openai_agent.is_available()}")
        print(f"Groq available: {groq_agent.is_available()}")
        print(f"Ollama available: {ollama_agent.is_available()}")
        
        # Now you can use these LLM instances with LangChain chains, prompts, etc.
        # For example:
        # from langchain.prompts import PromptTemplate
        # from langchain.chains import LLMChain
        # 
        # prompt = PromptTemplate(template="Tell me about {topic}", input_variables=["topic"])
        # chain = LLMChain(llm=openai_llm, prompt=prompt)
        # result = chain.run(topic="machine learning")
        
    except Exception as e:
        print(f"Error: {e}")