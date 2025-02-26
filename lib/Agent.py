import os
from pathlib import Path
from typing import Tuple

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


class Agent:

    def query(self, text, data: list[Tuple[Document, float]], path: str) -> str:
        pass


class OpenAIAgent(Agent):

    def __init__(self, path: str = "./tokens.yaml", model: str = "gpt-4o-mini"):
        with open(path, "r") as file:
            tokens = yaml.safe_load(file)
        self.model = model
        self.client = OpenAI(
            api_key=tokens["openai"],
        )

    def query(self, text, data: list[Tuple[Document, float]], path: str = "./prompt/generic_prompt.txt") -> str:
        with open(path, "r") as file:
            template = file.read()
        
        context = "\n\n---\n\n".join([f"{doc.page_content}\nID: {doc.metadata.get('id')}" for doc, _score in data])
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context, question=text)
        #print(prompt)
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        ).choices[0].message.content # TODO: return choices


if __name__ == "__main__":
    working_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent
    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(f"{working_directory}/tokens.yaml"),
        db_path=f"{working_directory}/data/db/open_ai_small"
    )
    message = "According to the annual report, what is the Operating margin (%) for Altech Chemicals Ltd  (within the last period or at the end of the last period)? If data is not available, return 'N/A'"
    data = repo.query(message)
    agent = OpenAIAgent(path=f"{working_directory}/tokens.yaml")
    resp = agent.query(message, data, f"{working_directory}/prompt/generic_prompt.txt")
    print("!!!!!")
    print(resp)
