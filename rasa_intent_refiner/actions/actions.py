from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import sys
import os
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api_client import APIClient
from output_handler import OutputHandler

api_client = APIClient()
output_handler = OutputHandler()

class ActionAskClarificationTextGeneration(Action):
    def name(self) -> Text:
        return "action_ask_clarification_text_generation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        topic = tracker.get_slot("topic")
        style = tracker.get_slot("style")
        length = tracker.get_slot("length")
        domain = tracker.get_slot("domain")

        clarification_message = "To clarify, you want me to generate text"

        if topic:
            clarification_message += f" about {topic}"
        if style:
            clarification_message += f" in the style of {style}"
        if length:
            clarification_message += f" with a length of {length}"
        if domain:
            clarification_message += f" for the domain of {domain}"

        clarification_message += ". Is that correct?"

        dispatcher.utter_message(text=clarification_message)

        return []

class ActionAskClarificationQuestionAnswering(Action):
    def name(self) -> Text:
        return "action_ask_clarification_question_answering"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        question = tracker.get_slot("question")
        domain = tracker.get_slot("domain")

        clarification_message = f"You want me to answer the question: {question}"

        if domain:
            clarification_message += f" for the domain of {domain}"

        clarification_message += ". Is that correct?"

        dispatcher.utter_message(text=clarification_message)

        return []

class ActionAskClarificationDataAnalysis(Action):
    def name(self) -> Text:
        return "action_ask_clarification_data_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        data_file = tracker.get_slot("data_file")
        domain = tracker.get_slot("domain")

        clarification_message = f"You want me to analyze the data file: {data_file}"

        if domain:
            clarification_message += f" for the domain of {domain}"

        clarification_message += ". Is that correct?"

        dispatcher.utter_message(text=clarification_message)

        return []

class ActionCallLLM(Action):
    def name(self) -> Text:
        return "action_call_llm"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        intent = tracker.latest_message['intent'].get('name')
        model_type = tracker.get_slot("model_type")
        stream_output = tracker.get_slot("stream_output") or False

        # Prepare prompt and parameters based on intent
        if intent == "text_generation":
            prompt = self._prepare_text_generation_prompt(tracker)
            task = "text_generation"
        elif intent == "question_answering":
            prompt = tracker.get_slot("question")
            task = "question_answering"
        elif intent == "data_analysis":
            prompt = f"Analyze this data: {tracker.get_slot('data_file')}"
            task = "data_analysis"
        else:
            dispatcher.utter_message(text="I'm not sure what you want me to do.")
            return []

        domain = tracker.get_slot("domain") or "education"

        try:
            if stream_output:
                # Handle streaming output
                llm_result = api_client.call_llm(prompt, model_type=model_type, task=task, domain=domain, stream=True)
            else:
                # Use ensemble approach for non-streaming requests
                llm_result = api_client.call_llm_with_ensemble(prompt, task=task, domain=domain)

            if isinstance(llm_result, dict) and llm_result.get("success", False):
                formatted_output = output_handler.format_output(llm_result["text"], format="plain_text", domain=domain)
                dispatcher.utter_message(text=formatted_output)
            else:
                return self._handle_error(dispatcher, tracker)
        except Exception as e:
            logger.error(f"Error in ActionCallLLM: {e}")
            return self._handle_error(dispatcher, tracker)

        return []

    def _prepare_text_generation_prompt(self, tracker: Tracker) -> str:
        topic = tracker.get_slot("topic")
        style = tracker.get_slot("style")
        length = tracker.get_slot("length")
        domain = tracker.get_slot("domain")
        
        prompt = f"Write a{' ' + length if length else ''} story about {topic}"
        if style:
            prompt += f" in the style of {style}"
        if domain:
            prompt += f" for {domain} purposes"
        return prompt

    def _handle_error(self, dispatcher: CollectingDispatcher, tracker: Tracker) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="I encountered an error. Please try again or rephrase your request.")
        return [FollowupAction("action_handle_error")]

class ActionHandleError(Action):
    def name(self) -> Text:
        return "action_handle_error"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Would you like to try with a different model or approach?")
        return []