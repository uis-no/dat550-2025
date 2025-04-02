"""A class for interacting with the Ollama API.

Usage:
    ollama = Ollama(config_path="code/llm_utils/mistral.yaml")
    ollama.generate("Write a story about a happy llama.")
"""

from ollama import Client, Options
from typing import Dict, Any, List
import yaml
import re
import os
from torch import Tensor

CHECKWORTHY_PROMPT = """Given a claim your task is to predict if it is check-worthy or not.{claim}"""
SENTIMENT_PROMPT = """Given a statement your task is to predict if it is positive, negative or neutral. Generate response as a JSON with key label and value positive, negative or neutral {claim}"""

class Ollama:
    """A class for generating questions and interacting with the Ollama API."""

    def __init__(self, config_path: str = "model.yaml"):
        """Initializes the Ollama client and loads necessary configurations."""
        self._ollama_client = Client(host="https://ollama.ux.uis.no", timeout=20)
        self._config_path = config_path
        self._config = self._load_config()
        self._stream = self._config.get("stream", False)
        self._model_name = self._config.get("model", "mistral")
        self._llm_options = self._get_llm_config()

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama LLM for the given prompt.

        Args:
            prompt: Prompt for the LLM.

        Returns:
            Response text from an Ollama LLM.
        """
        response = self._ollama_client.generate(
            model=self._model_name,
            prompt=prompt,
            options=self._llm_options,
            stream=self._stream,
            # format="json",
        )
        return response.get("response", "").strip()  # type: ignore

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from a YAML file.

        Raises:
            FileNotFoundError: If the config file is not found.

        Returns:
            A dictionary with configuration values.
        """
        if not os.path.isfile(self._config_path):
            raise FileNotFoundError(
                f"Config file {self._config_path} not found."
            )
        with open(self._config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data

    def _get_llm_config(self) -> Options:
        """Extracts and returns the LLM (language learning model) configuration.

        Returns:
            An Options object with the LLM configuration.
        """
        return Options(**self._config.get("options", {}))

if __name__ == "__main__":
    ollama = Ollama()
    # print(ollama.generate(CHECKWORTHY_PROMPT.format(claim="The earth is flat.")))
    # print(ollama.generate(SENTIMENT_PROMPT.format(claim="This movie is sickly good.")))
    print(ollama.generate("What do know about Stavanger?"))