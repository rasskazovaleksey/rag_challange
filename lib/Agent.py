import os
from pathlib import Path
from typing import Tuple

import requests
import yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


class Agent:

    def query(self, text, data: list[Tuple[Document, float]], path: str, system: str) -> str:
        pass


class IBMWatsonAgent(Agent):
    def __init__(self, path: str = "./tokens.yaml", model: str = "deepseek/deepseek-r1-distill-llama-70b"):
        try:
            with open(path, "r") as file:
                tokens = yaml.safe_load(file)
            self.token = tokens["watson"]
            if not self.token:
                raise ValueError("watson not found in tokens file.")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokens from {path}: {e}")

        self.model = model
        self.url = "https://rag.timetoact.at/ibm/text_generation"

    def query(self,
              text: str,
              data: list[Tuple[Document, float]],
              path: str = "./prompt/generic_prompt.txt",
              system: str = "You are a data extraction engine.",
              ) -> str:
        try:
            with open(path, "r") as file:
                template = file.read().strip()
        except Exception:
            ValueError(f"Failed to load prompt template from {path}")

        context = "\n\n---\n\n".join([f"{doc.page_content}" for doc, _score in data])
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context, question=text)

        # Build the payload using the prompt template and the provided text and context.
        payload = {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 4_000,
            },
            "model_id": self.model,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["results"][0]["generated_text"]
        except requests.HTTPError as err:
            return str(err)


class OpenAIAgent(Agent):

    def __init__(self, path: str = "./tokens.yaml", model: str = "gpt-4o-mini"):
        """
        NOTE: my teal doesn't allow o3-mini with reasoning_effort="high"
        :param path:
        :param model:
        """
        with open(path, "r") as file:
            tokens = yaml.safe_load(file)
        self.model = model
        self.llm = ChatOpenAI(
            api_key=tokens["openai"],
            model=self.model,
            # reasoning_effort="high",
        )

    def query(self, text, data: list[Tuple[Document, float]], path: str = "./prompt/generic_prompt.txt",
              system: str = "You are a data extraction engine.", ) -> str:
        with open(path, "r") as file:
            template = file.read()

        context = "\n\n---\n\n".join([f"{doc.page_content}\nID: {doc.metadata.get('id')}" for doc, _score in data])
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context, question=text, system="asdad")
        message = self.llm.invoke(prompt)
        return message.content


if __name__ == "__main__":
    working_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent
    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(f"{working_directory}/tokens.yaml"),
        db_path=f"{working_directory}/data/db/open_ai_small_50_10"
    )
    message = "According to the annual report, what is the Operating margin (%) for Altech Chemicals Ltd  (within the last period or at the end of the last period)? If data is not available, return 'N/A'"
    data = repo.query(message)

    agent = OpenAIAgent(path=f"{working_directory}/tokens.yaml")
    resp = agent.query(message, data, f"{working_directory}/prompt/number_prompt.txt")
    print("!!!!!")
    print(resp)

    # agent = IBMWatsonAgent(path=f"{working_directory}/tokens.yaml")
    # resp = agent.query(message, data, f"{working_directory}/prompt/number_prompt.txt")
    # print("!!!!!")
    # print(resp)
