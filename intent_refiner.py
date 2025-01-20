import tkinter as tk
from tkinter import simpledialog
from api_client import APIClient
from security import InputSanitizer
from typing import Callable, Optional

class IntentRefiner:
    """
    Refines user intents through iterative clarification using an API client.
    """
    def __init__(self, api_client: APIClient, max_iterations: int = 3):
        """
        Initializes the IntentRefiner with an APIClient and a maximum number of iterations.

        Args:
            api_client: The API client for making LLM calls.
            max_iterations: The maximum number of clarification iterations.
        """
        self.api_client = api_client
        self.max_iterations = max_iterations
        self._question_callback = None

    def set_question_callback(self, callback: Callable[[str], Optional[str]]):
        """Set callback for asking questions in GUI mode"""
        self._question_callback = callback

    def refine_intent(self, initial_input: str, task: str, domain: str) -> str:
        """
        Refines the user's intent through iterative clarification.
        Uses callback if in GUI mode, otherwise runs in console mode.

        Args:
            initial_input: The initial user input.
            task: The task type.
            domain: The domain context.

        Returns:
            The refined user input.
        """
        safe_input = InputSanitizer.sanitize(initial_input)
        refined_input = safe_input
        
        for i in range(self.max_iterations):
            question = self.api_client.call_llm(
                prompt=f"Clarify the following request: {refined_input}",
                task="question_generation",
                domain=domain,
            )["text"]

            if not question or "not clear" in question.lower():
                break

            # Use callback if available (GUI mode), otherwise use console input
            if self._question_callback:
                user_response = self._question_callback(question)
                if user_response is None:  # User cancelled
                    break
            else:
                print("\nClarifying question:", question)
                user_response = input("Your response (or press Enter to skip): ")
                if not user_response:
                    break
            
            safe_response = InputSanitizer.sanitize(user_response)
            refined_input = f"{refined_input}\nContext: {safe_response}"

        return refined_input