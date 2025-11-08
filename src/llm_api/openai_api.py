"""OpenAI API wrapper with retry logic."""
import json
import openai
from typing import Dict, List, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_exception_type
)

from .base import BaseLLMAPI, LLMAPIError, OutputParsingError


STOP_AFTER_ATTEMPT = 4


class OpenAIAPI(BaseLLMAPI):
    """OpenAI API wrapper with automatic retry on failures."""

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini"):
        """Initialize OpenAI API wrapper.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4')
        """
        super().__init__(api_key, model)
        if api_key:
            openai.api_key = api_key

    def convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """OpenAI uses messages as-is (no conversion needed).

        Args:
            messages: Standard message format

        Returns:
            Same messages (OpenAI format is the standard)
        """
        return messages

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for _ in range(2)] + [wait_fixed(5)]),
        retry=retry_if_exception_type((LLMAPIError, OutputParsingError, json.JSONDecodeError, Exception))
    )
    def complete(
        self,
        messages: List[Dict[str, str]],
        json_parsing_check: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with automatic retry.

        Args:
            messages: List of message dictionaries
            json_parsing_check: Whether to validate JSON response
            verbose: Whether to print debug information
            **kwargs: Additional OpenAI parameters (temperature, max_tokens, etc.)

        Returns:
            Response dictionary with 'choices' key

        Raises:
            LLMAPIError: If API call fails after retries
            OutputParsingError: If JSON parsing fails
        """
        if verbose:
            print(f"[OpenAI] Calling {self.model}")

        # Add JSON response format if needed
        if json_parsing_check:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **kwargs
            )

            if verbose:
                print(f"[OpenAI] Response: {response}")

            # Validate JSON if requested
            if json_parsing_check:
                self._validate_json_response(response)

            return response

        except Exception as e:
            if verbose:
                print(f"[OpenAI] Error: {e}")
            # Check for specific error types
            error_str = str(e)
            if "500" in error_str or "Internal Server Error" in error_str:
                print(f"[OpenAI] 500 Error encountered. Retrying...")
            raise LLMAPIError(f"OpenAI API error: {e}")

    def _validate_json_response(self, response: Dict[str, Any]) -> None:
        """Validate that response contains valid JSON.

        Args:
            response: API response dictionary

        Raises:
            OutputParsingError: If JSON is invalid
        """
        try:
            choices = response['choices']
            for choice in choices:
                json.loads(choice['message']['content'])
        except json.JSONDecodeError as e:
            raise OutputParsingError(f"Invalid JSON in response: {e}")
        except KeyError as e:
            raise OutputParsingError(f"Unexpected response format: {e}")
