import os

class PromptEngineer:
    """
    Generates prompts for LLMs based on templates and input parameters.
    """
    def __init__(self, prompt_dir: str = "prompts"):
        """
        Initializes the PromptEngineer with a directory containing prompt templates.

        Args:
            prompt_dir: The directory containing prompt templates.
        """
        self.prompt_dir = prompt_dir

    def generate_prompt(self, task: str, domain: str, input: str, **kwargs) -> str:
        """
        Generates a prompt based on the specified task, domain, and input.

        Args:
            task: The task type.
            domain: The domain context.
            input: The user input.
            **kwargs: Additional keyword arguments to be used as placeholders in the template.

        Returns:
            The generated prompt.
        """
        template_path = os.path.join(self.prompt_dir, task, domain, "template.txt")
        try:
            with open(template_path, "r") as f:
                template = f.read()
        except FileNotFoundError:
            # Fallback to basic template if specific one not found
            default_template = (
                f"Task: {task}\n"
                f"Domain: {domain}\n"
                f"Request: {{input}}\n\n"
                "Please provide a detailed and well-structured response."
            )
            print(f"Warning: Prompt template not found at {template_path}. Using default template.")
            template = default_template

        # Replace placeholders with values
        prompt = template.replace("{{input}}", input)
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        return prompt

    def get_model_for_task(self, task: str, domain: str) -> str:
        """
        Selects the appropriate model based on task and domain requirements.
        """
        config = self._load_model_config()
        
        # Check domain-specific preferred model
        domain_config = config.get("domain_defaults", {}).get(domain, {})
        if "preferred_model" in domain_config:
            return domain_config["preferred_model"]
        
        # Fall back to task-specific default model
        task_config = config.get("tasks", {}).get(task, {})
        return task_config.get("default", ["hf:mistralai/Mistral-Nemo-Instruct-2407"])[0]