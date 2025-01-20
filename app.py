from prefect import task, flow
from api_client import APIClient
from intent_refiner import IntentRefiner
from output_handler import OutputHandler

api_client = APIClient()
intent_refiner = IntentRefiner(api_client)
output_handler = OutputHandler()

@task
def refine_intent_task(initial_input: str, task: str, domain: str):
    return intent_refiner.refine_intent(initial_input, task, domain)

@task
def call_llm_task(prompt: str, model_type: str = None, task: str = None, domain: str = None, stream: bool = False):
    return api_client.call_llm_with_ensemble(prompt, task, domain)

@task
def format_output_task(output: str, format: str = "plain_text", domain: str = None):
    return output_handler.format_output(output, format, domain)

@flow(name="AI Agent Workflow")
def my_workflow(initial_input: str, task: str, domain: str, output_format: str = "plain_text"):
    refined_input = refine_intent_task(initial_input, task, domain)
    llm_result = call_llm_task(refined_input, task=task, domain=domain)
    formatted_output = format_output_task(llm_result, format=output_format, domain=domain)
    output_handler.present_output(formatted_output)
    return formatted_output

if __name__ == "__main__":
    initial_input = input("Enter your request: ")
    task = input("Enter the task (e.g., text_generation, question_answering): ")
    domain = input("Enter the domain (e.g., education, marketing): ")
    output_format = input("Enter the output format (e.g., plain_text, markdown, json, html): ")

    result = my_workflow(initial_input, task, domain, output_format)