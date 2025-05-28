import os
from pathlib import Path
from typing import Tuple

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


class Agent:

    def query(self, text, data: list[Tuple[Document, float]], path: str, system: str) -> str:
        pass


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
