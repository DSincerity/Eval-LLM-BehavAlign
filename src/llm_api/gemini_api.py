"""Google Gemini API wrapper with retry logic."""
import os
import json
from typing import Dict, List, Any, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_exception_type
)
from google import genai
from google.genai import types

from .base import BaseLLMAPI, LLMAPIError, OutputParsingError


STOP_AFTER_ATTEMPT = 4


class GeminiAPI(BaseLLMAPI):
    """Google Gemini API wrapper with automatic retry on failures."""

    def __init__(self, api_key: str = "", model: str = "gemini-2.0-flash"):
        """Initialize Gemini API wrapper.

        Args:
            api_key: Google API key (defaults to GEMINI_API_KEY or gemini_key env var)
            model: Model name
        """
        super().__init__(api_key, model)
        # Support both GEMINI_API_KEY (standard) and gemini_key (legacy)
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('gemini_key', '')
        self.client = genai.Client(api_key=self.api_key)

    def convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Convert OpenAI-style messages to Gemini format.

        Gemini uses 'model' role instead of 'assistant' and requires 'parts' structure.

        Args:
            messages: Standard message format with 'system', 'user', 'assistant' roles

        Returns:
            Tuple of (system_instruction, gemini_messages)
        """
        system_instruction = ""
        gemini_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
            else:
                raise ValueError(f"Unknown role: {msg['role']}")

        return system_instruction, gemini_messages

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
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with automatic retry.

        Args:
            messages: List of message dictionaries
            json_parsing_check: Whether to validate JSON response
            verbose: Whether to print debug information
            n: Number of completions to generate
            **kwargs: Additional Gemini parameters

        Returns:
            Response dictionary in OpenAI-compatible format

        Raises:
            LLMAPIError: If API call fails after retries
            OutputParsingError: If JSON parsing fails
        """
        if verbose:
            print(f"[Gemini] Calling {self.model}")

        # Convert messages to Gemini format
        system_instruction, gemini_messages = self.convert_messages(messages)

        # Build generation config
        gen_config = {
            "candidate_count": n
        }

        if system_instruction:
            gen_config["system_instruction"] = system_instruction

        if json_parsing_check:
            gen_config["response_mime_type"] = "application/json"

        if verbose:
            print(f"[Gemini] System: {system_instruction}")
            print(f"[Gemini] Messages: {gemini_messages}")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**gen_config),
                contents=gemini_messages
            )

            # Convert to OpenAI-compatible format
            formatted_response = self._format_response(response)

            if verbose:
                print(f"[Gemini] Response: {formatted_response}")

            # Validate JSON if requested
            if json_parsing_check:
                self._validate_json_response(formatted_response)

            return formatted_response

        except Exception as e:
            if verbose:
                print(f"[Gemini] Error: {e}")
            raise LLMAPIError(f"Gemini API error: {e}")

    def _format_response(self, response: Any) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI-compatible format.

        Args:
            response: Gemini API response

        Returns:
            OpenAI-style response dictionary
        """
        formatted = {"choices": []}

        for candidate in response.candidates:
            # Clean markdown code blocks from JSON responses
            text = candidate.content.parts[0].text
            text = text.replace("json", "").replace("`", "")

            formatted["choices"].append({
                'message': {
                    "role": "assistant",
                    "content": text
                }
            })

        return formatted

    def _validate_json_response(self, response: Dict[str, Any]) -> None:
        """Validate that response contains valid JSON.

        Args:
            response: Formatted response dictionary

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
