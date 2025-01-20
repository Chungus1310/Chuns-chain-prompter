# Chun's Chain Prompter

Welcome to **Chun's Chain Prompter**! 🚀 This project is designed to help you generate high-quality text content, answer questions, and perform data analysis using advanced language models for *FREE*. Whether you're a developer, content creator, or data analyst, this tool is here to make your life easier by automating complex workflows and refining your prompts for better results.

---

## 🌟 Features

- **Prompt Refinement**: Automatically refine your input prompts to get better results from language models.
- **Multi-Model Support**: Supports multiple language models including Hugging Face, OpenRouter, and Google's Gemini.
- **Task-Specific Output**: Generate text, answer questions, or analyze data with domain-specific configurations.
- **GUI Interface**: A user-friendly graphical interface for easy interaction.
- **Chatbot Integration**: A chatbot interface for conversational interactions.
- **Customizable Prompts**: Easily customize prompts for different tasks and domains.
- **Output Formatting**: Format outputs in plain text, markdown, JSON, or HTML.
- **API Fallback System**: Ensures reliability by automatically switching to fallback models if the primary model fails.
- **Rasa Integration**: Optional Rasa-based intent refinement for more advanced conversational flows.

> **Note**: The **data analysis** feature is currently a **work in progress** and does not yet support file uploads. Stay tuned for updates! 🛠️  
> **Note**: The **Rasa integration** is available in both the GUI and CLI versions but has not been fully tested yet. Use with caution and feel free to contribute! 🛠️

---

## 📂 Project Structure

Here’s a detailed overview of the project structure and what each file/folder does:

### **Root Directory**
- **`api_client.py`**: The core client for interacting with various LLM APIs (Hugging Face, OpenRouter, Gemini). It includes features like rate limiting, retries, and the **API fallback system**.
- **`intent_refiner.py`**: Refines user input by asking clarifying questions and ensuring the input is clear and concise. Works with both GUI and CLI.
- **`output_handler.py`**: Handles and formats the output from LLMs into plain text, markdown, JSON, or HTML.
- **`prompt_engineer.py`**: Generates prompts based on predefined templates for different tasks and domains.
- **`app.py`**: The main CLI application that runs the workflow. It supports Rasa integration for intent refinement.
- **`gui.py`**: A graphical user interface for the project, built with `tkinter`. It also supports Rasa integration.
- **`chatbot.py`**: A simple chatbot interface using `chatterbot` for conversational interactions.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`model_config.json`**: Configuration file for model selection, task-specific settings, and domain defaults.
- **`security.py`**: Handles input sanitization to prevent malicious input.
- **`validation.py`**: Validates LLM outputs based on domain-specific rules (e.g., minimum length, blacklisted terms).
- **`.env`**: Environment variables file (not included in the repo) for storing API keys and sensitive data.

---

### **`prompts/`**
This folder contains prompt templates for different tasks and domains. Each task (e.g., text generation, question answering) has its own subfolder with domain-specific templates.

- **`text_generation/`**: Templates for generating text in various domains like education, marketing, and data analysis.
- **`question_generation/`**: Templates for generating clarifying questions to refine user input.

---

### **`rasa_intent_refiner/`**
This folder contains the Rasa-based intent refinement system. It’s an optional feature for more advanced conversational flows.

- **`actions/`**: Contains custom Rasa actions, including the `action_call_llm` action for interacting with LLMs.
- **`data/`**: Contains Rasa training data (`nlu.yml` for intents and `stories.yml` for conversation flows).
- **`config.yml`**: Rasa configuration file for the NLU pipeline and policies.
- **`domain.yml`**: Defines the Rasa domain, including intents, entities, slots, and responses.
- **`endpoints.yml`**: Configures the Rasa action server endpoint.
- **`credentials.yml`**: Stores credentials for Rasa (not included in the repo).

---

### **API Fallback System**
The **API fallback system** is a key feature of the `api_client.py` file. It ensures reliability by automatically switching to fallback models if the primary model fails. Here’s how it works:

1. **Primary Model**: The system first tries to use the primary model specified in `model_config.json` for the given task.
2. **Fallback Models**: If the primary model fails (e.g., due to rate limits or errors), the system automatically switches to fallback models in the order specified in the configuration.
3. **Ensembling**: If multiple models succeed, the system uses a **majority voting mechanism** to select the best output.

This system ensures that the application remains functional even if one or more APIs are unavailable or fail.

---

## 🛠️ Installation

To get started with Chun's Chain Prompter, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Chungus1310/Chuns-chain-prompter.git
   cd Chuns-chain-prompter
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```plaintext
   GEMINI_API_KEY=your_gemini_api_key
   HF_API_KEY=your_huggingface_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

---

## 🚀 Usage

### Running the GUI
To launch the graphical user interface, run:
```bash
python gui.py
```

### Running the Chatbot
To start the chatbot, run:
```bash
python chatbot.py
```

### Running the CLI
For command-line interaction, use:
```bash
python app.py
```

### Rasa Integration (Optional)
The Rasa integration is available in both the GUI (`gui.py`) and CLI (`app.py`) versions. However, it has not been fully tested yet. If you'd like to experiment with it, ensure you have Rasa installed and configured properly.

---

## 🤖 How It Works

1. **Input Refinement**: The `IntentRefiner` class refines your input prompts by asking clarifying questions and ensuring the input is clear and concise.
2. **Prompt Engineering**: The `PromptEngineer` class generates prompts based on predefined templates for different tasks and domains.
3. **Model Selection**: The `APIClient` class selects the appropriate language model based on the task and domain, with fallback options if the primary model fails.
4. **Output Handling**: The `OutputHandler` class formats the output based on your preferences (plain text, markdown, JSON, or HTML).
5. **Rasa Integration**: Optional Rasa-based intent refinement for more advanced conversational flows (available but not fully tested).

---

## 📝 Customization

You can customize the prompts by editing the template files in the `prompts/` directory. Each task and domain has its own template, allowing you to tailor the prompts to your specific needs.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributing

We welcome contributions! If you have any ideas, bug fixes, or improvements, feel free to open an issue or submit a pull request. Please make sure to follow the [contribution guidelines](CONTRIBUTING.md).

---

## 📧 Contact

If you have any questions or need help, feel free to reach out to me on [GitHub](https://github.com/Chungus1310).

---

Happy prompting! 🎉
