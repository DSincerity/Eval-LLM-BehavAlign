"""Anthropic Claude API wrapper with retry logic."""
import os
import json
import anthropic
from typing import Dict, List, Any, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_exception_type
)

from .base import BaseLLMAPI, LLMAPIError, OutputParsingError


STOP_AFTER_ATTEMPT = 4


class ClaudeAPI(BaseLLMAPI):
    """Anthropic Claude API wrapper with automatic retry on failures."""

    def __init__(self, api_key: str = "", model: str = "claude-3-7-sonnet-20250219"):
        """Initialize Claude API wrapper.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY or anthropic_key env var)
            model: Model name
        """
        super().__init__(api_key, model)
        # Support both ANTHROPIC_API_KEY (standard) and anthropic_key (legacy)
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '') or os.environ.get('anthropic_key', '')
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Convert OpenAI-style messages to Claude format.

        Claude requires system messages to be separate from conversation messages.

        Args:
            messages: Standard message format with 'system', 'user', 'assistant' roles

        Returns:
            Tuple of (system_instruction, conversation_messages)
        """
        system_instruction = ""
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                conversation_messages.append(msg)

        return system_instruction, conversation_messages

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
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with automatic retry.

        Args:
            messages: List of message dictionaries
            json_parsing_check: Whether to validate JSON response
            verbose: Whether to print debug information
            max_tokens: Maximum tokens in response
            **kwargs: Additional Claude parameters

        Returns:
            Response dictionary in OpenAI-compatible format

        Raises:
            LLMAPIError: If API call fails after retries
            OutputParsingError: If JSON parsing fails
        """
        if verbose:
            print(f"[Claude] Calling {self.model}")

        # Convert messages to Claude format
        system_instruction, conversation_messages = self.convert_messages(messages)

        if verbose:
            print(f"[Claude] System: {system_instruction}")
            print(f"[Claude] Messages: {conversation_messages}")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_instruction,
                messages=conversation_messages,
                **kwargs
            )

            # Convert to OpenAI-compatible format
            formatted_response = self._format_response(response)

            if verbose:
                print(f"[Claude] Response: {formatted_response}")

            # Validate JSON if requested
            if json_parsing_check:
                self._validate_json_response(formatted_response)

            return formatted_response

        except Exception as e:
            if verbose:
                print(f"[Claude] Error: {e}")
            raise LLMAPIError(f"Claude API error: {e}")

    def _format_response(self, response: Any) -> Dict[str, Any]:
        """Convert Claude response to OpenAI-compatible format.

        Args:
            response: Claude API response

        Returns:
            OpenAI-style response dictionary
        """
        formatted = {"choices": []}

        for content_block in response.content:
            formatted["choices"].append({
                'message': {
                    "role": "assistant",
                    "content": content_block.text
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
