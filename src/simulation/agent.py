"""Dialog agent implementation for L2L negotiations."""
from copy import deepcopy
from typing import List, Dict, Optional

from ..llm_api import get_llm_api, BaseLLMAPI


class DialogAgent:
    """Dialog Agent for negotiation scenarios.

    Args:
        initial_dialog_history: Initial conversation history
        agent_type: Type/personality of the agent
        engine: LLM engine name (e.g., 'gpt-4o-mini', 'claude-3-7-sonnet')
        api_key: API key for the LLM service
        system_instruction: System instruction/prompt for the agent
    """

    def __init__(
        self,
        initial_dialog_history: Optional[List[Dict[str, str]]] = None,
        agent_type: str = "",
        engine: str = "gpt-4o-mini",
        api_key: str = "",
        system_instruction: Optional[str] = None
    ):
        """Initialize the dialog agent."""
        self.agent_type = agent_type
        self.engine = engine
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.last_prompt = ""

        # Get appropriate LLM API wrapper
        self.llm_api: BaseLLMAPI = get_llm_api(engine, api_key)

        # Initialize dialog history
        self.initialize_agent(initial_dialog_history, system_instruction)

        print(f"[DialogAgent] Initializing {self.agent_type} with engine ({self.engine})")

    def initialize_agent(
        self,
        initial_dialog_history: Optional[List[Dict[str, str]]],
        system_instruction: Optional[str]
    ) -> None:
        """Initialize the dialog history.

        Args:
            initial_dialog_history: Initial dialog history if provided
            system_instruction: System instruction to use if no initial history

        Raises:
            AssertionError: If system instruction is None when no initial history
        """
        if initial_dialog_history is None:
            assert system_instruction is not None, \
                "System instruction must be provided if no initial dialog history is given."
            self.dialog_history = [{"role": "system", "content": system_instruction}]
            self.initial_dialog_history = [{"role": "system", "content": system_instruction}]
        else:
            self.dialog_history = deepcopy(initial_dialog_history)
            self.initial_dialog_history = deepcopy(initial_dialog_history)

    def reset(self) -> None:
        """Reset dialog history to initial state."""
        self.dialog_history = deepcopy(self.initial_dialog_history) if self.initial_dialog_history else []

    def call(
        self,
        prompt: str,
        only_w_system_instruction: bool = False,
        **kwargs
    ) -> str:
        """Call the agent with a prompt.

        Args:
            prompt: The prompt to send to the agent
            only_w_system_instruction: If True, only use system instruction
            **kwargs: Additional parameters for the LLM API

        Returns:
            The response content from the agent
        """
        if not only_w_system_instruction:
            prompt_msg = {"role": "user", "content": prompt}
            self.dialog_history.append(prompt_msg)
            self.last_prompt = prompt

        # Call LLM API
        messages = list(self.dialog_history)
        response = self.llm_api.complete(messages, **kwargs)

        # Extract message from response
        message = response['choices'][0]['message']
        assert message['role'] == 'assistant', f"Unexpected role: {message['role']}"

        # Add to dialog history
        self.dialog_history.append(dict(message))

        return message['content']

    @property
    def last_response(self) -> str:
        """Get the last response content from dialog history."""
        return self.dialog_history[-1]['content']

    @property
    def history(self) -> None:
        """Print the dialog history."""
        for h in self.dialog_history:
            print(f'{h["role"]}:  {h["content"]}')
