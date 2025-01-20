import os
import json
from typing import Dict, Any
import requests
import google.generativeai as genai
from huggingface_hub import InferenceClient
from prompt_engineer import PromptEngineer
import logging
import functools
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime, timedelta
from circuitbreaker import circuit
import time

load_dotenv()

# Configure logging
logging.basicConfig(filename="app.log", level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class APIClient:
    """
    Client for interacting with various Large Language Model APIs.
    Supports Hugging Face, OpenRouter, and Google's Gemini, with features like:
    - Automatic model selection based on task and domain.
    - Fallback to alternative models.
    - Rate limiting.
    - Retries with exponential backoff.
    - Circuit breaker pattern for error handling.
    - Model ensembling through majority voting.
    """
    def __init__(self):
        """
        Initializes the APIClient, configuring Gemini and setting up prompt engineering.
        """
        self.hf_client = None
        self.gemini_client = None
        self.gemini_generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self._configure_gemini()
        self.prompt_engineer = PromptEngineer()
        self.rate_limits = {
            "hf": {"last_call": None, "count": 0},
            "gemini": {"last_call": None, "count": 0},
            "openrouter": {"last_call": None, "count": 0}
        }

    def _configure_gemini(self):
        """
        Configures the Gemini API client using the API key from environment variables.
        """
        try:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.gemini_client = genai.GenerativeModel(
                model_name="gemini-1.5-flash-8b",
                generation_config=self.gemini_generation_config,
            )
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")

    def _check_rate_limit(self, provider: str) -> bool:
        """
        Checks if the rate limit for a given API provider has been exceeded.

        Args:
            provider: The API provider (hf, gemini, openrouter).

        Returns:
            True if the rate limit has not been exceeded, False otherwise.
        """
        limit = {
            "hf": (30, 60),  # 30 calls/minute
            "gemini": (60, 60), # 60 calls/minute
            "openrouter": (20, 60) # 20 calls/minute
        }.get(provider, (10, 60))
        
        now = datetime.now()

        if self.rate_limits[provider]["last_call"] is None:
            self.rate_limits[provider]["last_call"] = now
            self.rate_limits[provider]["count"] = 0

        if (now - self.rate_limits[provider]["last_call"]) < timedelta(seconds=limit[1]):
            if self.rate_limits[provider]["count"] >= limit[0]:
                logger.warning(f"Rate limit exceeded for {provider}")
                return False
            else:
                self.rate_limits[provider]["count"] += 1
        else:
            self.rate_limits[provider]["last_call"] = now
            self.rate_limits[provider]["count"] = 1
        return True

    def _get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific configuration parameters."""
        config = self._load_model_config()
        return config.get("domain_defaults", {}).get(domain, {})

    @functools.lru_cache(maxsize=128)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    @circuit(failure_threshold=3, recovery_timeout=30)
    def _call_huggingface(self, prompt: str, model: str, domain: str = None, stream: bool = False) -> Dict[str, Any]:
        """
        Calls the Hugging Face API to generate text.

        Args:
            prompt: The input prompt.
            model: The Hugging Face model name.
            domain: The domain context.
            stream: Whether to use streaming.

        Returns:
            A dictionary containing the generated text, model name, success status, and any error message.
        """
        if not self._check_rate_limit("hf"):
            return {"text": "", "model": model, "success": False, "error": "Rate limit exceeded for Hugging Face API."}
        try:
            if not self.hf_client:
                self.hf_client = InferenceClient(api_key=os.environ.get("HF_API_KEY"))

            # Get domain-specific parameters
            domain_config = self._get_domain_config(domain)
            temperature = domain_config.get("temperature", 0.5)
            max_tokens = domain_config.get("max_tokens", 2048)

            messages = [{"role": "user", "content": prompt}]
            if stream:
                stream_response = self.hf_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.7,
                    stream=True
                )
                full_text = ""
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_text += chunk.choices[0].delta.content
                return {"text": full_text, "model": model, "success": True}
            else:
                response = self.hf_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.7
                )
                return {"text": response.choices[0].message.content, "model": model, "success": True}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            return {"text": "", "model": model, "success": False, "error": "Error calling Hugging Face API. Check logs for details."}

    @functools.lru_cache(maxsize=128)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    @circuit(failure_threshold=3, recovery_timeout=30)
    def _call_openrouter(self, prompt: str, model: str, domain: str = None) -> Dict[str, Any]:
        """
        Calls the OpenRouter API to generate text.

        Args:
            prompt: The input prompt.
            model: The OpenRouter model name.
            domain: The domain context.

        Returns:
            A dictionary containing the generated text, model name, success status, and any error message.
        """
        if not self._check_rate_limit("openrouter"):
            return {"text": "", "model": model, "success": False, "error": "Rate limit exceeded for OpenRouter API."}
        try:
            # Get domain-specific parameters
            domain_config = self._get_domain_config(domain)
            temperature = domain_config.get("temperature", 0.7)
            max_tokens = domain_config.get("max_tokens", 2048)

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }),
            )
            response.raise_for_status()
            return {"text": response.json()["choices"][0]["message"]["content"], "model": model, "success": True}
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout error for {model}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding OpenRouter API response: {e}")
            return {"text": "", "model": model, "success": False, "error": "Error decoding OpenRouter API response. Check logs for details."}

    @functools.lru_cache(maxsize=128)
    @circuit(failure_threshold=3, recovery_timeout=30)
    def _call_gemini(self, prompt: str, model_name: str, domain: str = None) -> Dict[str, Any]:
        """
        Calls the Gemini API to generate text.

        Args:
            prompt: The input prompt.
            model_name: The Gemini model name.
            domain: The domain context.

        Returns:
            A dictionary containing the generated text, model name, success status, and any error message.
        """
        if not self._check_rate_limit("gemini"):
            return {"text": "", "model": model_name, "success": False, "error": "Rate limit exceeded for Gemini API."}
        try:
            # Get domain-specific parameters
            domain_config = self._get_domain_config(domain)
            temperature = domain_config.get("temperature", 0.7)
            max_tokens = domain_config.get("max_tokens", 8192)

            if not self.gemini_client:
                self._configure_gemini()

            # Update generation config with domain-specific parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            self.gemini_client.generation_config = generation_config
            chat_session = self.gemini_client.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return {"text": response.text, "model": model_name, "success": True}
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {"text": "", "model": model_name, "success": False, "error": "Error calling Gemini API. Check logs for details."}

    def _load_model_config(self, config_path: str = "model_config.json") -> Dict[str, Any]:
        """
        Loads the model configuration from a JSON file.

        Args:
            config_path: The path to the configuration file.

        Returns:
            A dictionary containing the model configuration.
        """
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Model configuration file not found at {config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in model configuration file at {config_path}")
            return {}
    
    def call_llm_with_ensemble(self, prompt: str, task: str, domain: str) -> str:
        """
        Calls LLMs with ensembling, using a primary model and fallbacks, and returns a consensus response.

        Args:
            prompt: The input prompt.
            task: The task type.
            domain: The domain context.

        Returns:
            The ensembled response text, or an error message if all models fail.
        """
        responses = []
        model_config = self._load_model_config()
        
        # Use configured primary model
        primary_model = model_config.get("model_priorities", {}).get(task, {}).get("primary", "gemini:gemini-1.5-flash")
        model_type, model_name = primary_model.split(":", 1)
        
        # Generate prompt using PromptEngineer
        final_prompt = self.prompt_engineer.generate_prompt(task, domain, prompt)

        if model_type == "hf":
            result = self._call_huggingface(final_prompt, model_name, domain)
        elif model_type == "openrouter":
            result = self._call_openrouter(final_prompt, model_name, domain)
        elif model_type == "gemini":
            result = self._call_gemini(final_prompt, model_name, domain)
        else:
            result = {"text": "", "model": "", "success": False, "error": f"Invalid model type: {model_type}"}

        if result["success"]:
            responses.append(result["text"])

        # Use configured fallback models if primary fails
        if not result["success"]:
            fallback_models = model_config.get("model_priorities", {}).get(task, {}).get("fallbacks", [])
            for model_str in fallback_models:
                model_type, model_name = model_str.split(":", 1)
                if model_type == "hf":
                    result = self._call_huggingface(final_prompt, model_name, domain)
                elif model_type == "openrouter":
                    result = self._call_openrouter(final_prompt, model_name, domain)
                elif model_type == "gemini":
                    result = self._call_gemini(final_prompt, model_name, domain)

                if result["success"]:
                    responses.append(result["text"])
                    break

        if len(responses) >= 2:
            return self._majority_vote(responses)
        return responses[0] if responses else "Error: All models failed"

    def _majority_vote(self, responses: list) -> str:
        """
        Implements a simple majority voting mechanism for model ensembling.

        Args:
            responses: A list of response texts.

        Returns:
            The response text that appears most frequently in the list.
        """
        # Implement consensus logic
        return max(set(responses), key=responses.count)

    def call_llm(self, prompt: str, model_type: str = None, task: str = None, domain: str = None, stream: bool = False) -> Dict[str, Any]:
        """
        Orchestrates LLM calls with automatic model selection, fallback, and prompt engineering.

        Args:
            prompt: The input prompt.
            model_type: The specific model type to use (optional).
            task: The task type.
            domain: The domain context.
            stream: Whether to use streaming (if available).

        Returns:
            A dictionary containing the generated text, model name, success status, and any error message.
        """
        if not model_type:
            # Try HuggingFace models first
            hf_models = ["mistralai/Mistral-Nemo-Instruct-2407", "01-ai/Yi-1.5-34B-Chat"]
            for model in hf_models:
                result = self._call_huggingface(prompt, model, domain, stream)
                if result["success"]:
                    return result

            # Try OpenRouter models next
            or_models = ["mistralai/mistral-7b-instruct:free", "google/gemma-2-9b-it:free"]
            for model in or_models:
                result = self._call_openrouter(prompt, model, domain)
                if result["success"]:
                    return result

            # Finally try Gemini
            result = self._call_gemini(prompt, "gemini-1.5-flash", domain)
            if result["success"]:
                return result

            return {"text": "", "model": "", "success": False, "error": "All models failed"}
        else:
            # Handle specific model type requests
            if model_type == "hf":
                models = ["mistralai/Mistral-Nemo-Instruct-2407", "01-ai/Yi-1.5-34B-Chat"]
            elif model_type == "openrouter":
                models = ["mistralai/mistral-7b-instruct:free", "google/gemma-2-9b-it:free"]
            elif model_type == "gemini":
                models = ["gemini-1.5-flash"]
            else:
                return {"text": "", "model": "", "success": False, "error": "Invalid model type"}

            for model in models:
                result = getattr(self, f"_call_{model_type}")(prompt, model, domain, stream if model_type == "hf" else False)
                if result["success"]:
                    return result

            return {"text": "", "model": "", "success": False, "error": f"All {model_type} models failed"}

    def benchmark_models(self, prompt: str) -> dict:
        """
        Benchmarks the performance of different LLM providers.

        Args:
            prompt: The input prompt to use for benchmarking.

        Returns:
            A dictionary containing the latency, success status, and response length for each provider.
        """
        results = {}
        for provider in ["gemini", "hf", "openrouter"]:
            start = time.time()
            response = self.call_llm(prompt, model_type=provider)
            results[provider] = {
                "latency": time.time() - start,
                "success": response["success"],
                "length": len(response["text"])
            }
        return results