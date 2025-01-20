from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from api_client import APIClient
from intent_refiner import IntentRefiner
from output_handler import OutputHandler
from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement

# Initialize components
api_client = APIClient()
intent_refiner = IntentRefiner(api_client)
output_handler = OutputHandler()

# Create chatbot
chatbot = ChatBot(
    'My ChatBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatbot.CustomLogicAdapter',
            'api_client': api_client,
            'intent_refiner': intent_refiner,
            'output_handler': output_handler
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

# Train chatbot (basic example)
trainer = ListTrainer(chatbot)
trainer.train([
    "Hi",
    "Hello there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
])

# Example of a custom logic adapter to handle workflow execution
class CustomLogicAdapter(LogicAdapter):
    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)
        self.api_client = kwargs['api_client']
        self.intent_refiner = kwargs['intent_refiner']
        self.output_handler = kwargs['output_handler']

    def can_process(self, statement):
        # Check if the statement can be processed by this adapter
        return True  # Process all statements for this example

    def process(self, input_statement, additional_response_selection_parameters):

        # Basic intent detection (replace with more sophisticated NLU)
        if "write a story" in input_statement.text.lower():
            task = "text_generation"
        elif "answer a question" in input_statement.text.lower():
            task = "question_answering"
        else:
            task = "unknown"

        domain = "education"  # Default domain for this example

        # Refine intent
        refined_input = self.intent_refiner.refine_intent(input_statement.text, task, domain)

        # Call LLM
        llm_result = self.api_client.call_llm_with_ensemble(refined_input, task=task, domain=domain)

        # Format output
        formatted_output = self.output_handler.format_output(llm_result, format="plain_text", domain=domain)

        # Create response statement
        response_statement = Statement(text=formatted_output)
        response_statement.confidence = 1  # Confidence score

        return response_statement

# Run chatbot
while True:
    try:
        user_input = input("You: ")
        response = chatbot.get_response(user_input)
        print("Bot: ", response)
    except (KeyboardInterrupt, EOFError, SystemExit):
        break