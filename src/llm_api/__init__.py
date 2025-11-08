"""LLM API wrappers for OpenAI, Claude, and Gemini."""
from .base import BaseLLMAPI, LLMAPIError, OutputParsingError
from .openai_api import OpenAIAPI
from .claude_api import ClaudeAPI
from .gemini_api import GeminiAPI


def get_llm_api(engine_name: str, api_key: str = "") -> BaseLLMAPI:
    """Factory function to get appropriate LLM API wrapper.

    Args:
        engine_name: Name of the engine (e.g., 'gpt-4o-mini', 'claude-3-7-sonnet', 'gemini-2.0-flash')
        api_key: API key for authentication

    Returns:
        Instance of appropriate API wrapper

    Raises:
        ValueError: If engine name is not recognized
    """
    engine_lower = engine_name.lower()

    if "gpt" in engine_lower:
        return OpenAIAPI(api_key=api_key, model=engine_name)
    elif "claude" in engine_lower:
        return ClaudeAPI(api_key=api_key, model=engine_name)
    elif "gemini" in engine_lower:
        return GeminiAPI(api_key=api_key, model=engine_name)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


__all__ = [
    'BaseLLMAPI',
    'LLMAPIError',
    'OutputParsingError',
    'OpenAIAPI',
    'ClaudeAPI',
    'GeminiAPI',
    'get_llm_api'
]
