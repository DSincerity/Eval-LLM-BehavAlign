"""Base classes and utilities for LLM API interactions."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class LLMAPIError(Exception):
    """Base exception for LLM API errors."""
    pass


class OutputParsingError(Exception):
    """Exception raised when output parsing fails."""
    pass


class BaseLLMAPI(ABC):
    """Abstract base class for LLM API wrappers."""

    def __init__(self, api_key: str = "", model: str = ""):
        """Initialize the API wrapper.

        Args:
            api_key: API key for authentication
            model: Model identifier
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional model-specific parameters

        Returns:
            Response dictionary with 'choices' key

        Raises:
            LLMAPIError: If API call fails
            OutputParsingError: If response parsing fails
        """
        pass

    @abstractmethod
    def convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> Any:
        """Convert messages to provider-specific format.

        Args:
            messages: Standard OpenAI-style messages

        Returns:
            Provider-specific message format
        """
        pass
