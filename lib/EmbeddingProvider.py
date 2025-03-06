import os
from pathlib import Path

import yaml
from ibm_watsonx_ai import APIClient, Credentials, href_definitions
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingProvider():
    def provide(self) -> Embeddings:
        pass


class OpenAiEmbeddingProvider(EmbeddingProvider):

    def __init__(self, path: str = "tokens.yaml", model: str = "text-embedding-3-small") -> None:
        with open(path, "r") as file:
            tokens = yaml.safe_load(file)
        self.token = tokens["openai"]
        self.model = model

    def provide(self) -> Embeddings:
        embeddings = OpenAIEmbeddings(
            model=self.model,
            api_key=self.token,
        )
        return embeddings


class WatsonEmbeddingProvider(EmbeddingProvider):

    def __init__(self, path: str = "tokens.yaml", model: str = "ibm/granite-embedding-107m-multilingual") -> None:
        """
        slate-125m-english-rtrvr - 768
        slate-30m-english-rtrvr - 384
        granite-embedding-107m-multilingual - 384
        """
        with open(path, "r") as file:
            tokens = yaml.safe_load(file)
        self.token = tokens["watson"]
        self.model = model
        self.api_client = APIClient(
            credentials=Credentials(
                url="https://rag.timetoact.at/ibm/embeddings",
                token=self.token,
                instance_id="openshift",
                version="5.0",
                username="alex",

            ),
        )
        href_definitions.FM_EMBEDDINGS = "{}"

    def provide(self) -> Embeddings:
        # embeddings = WatsonxEmbeddings(
        #     url="https://eu-de.ml.cloud.ibm.com",
        #     project_id="11326486-db97-45b5-961b-9211b14cf4c6",
        #     model_id=self.model,
        #     apikey=self.token,
        # )

        embeddings = CustomWatsonEmbeddings(
            token=self.token,
            model_id=self.model,
        )
        return embeddings


import requests


class CustomWatsonEmbeddings(Embeddings):
    def __init__(self, token: str, model_id: str = "ibm/granite-embedding-107m-multilingual"):
        self.token = token
        self.model_id = model_id
        self.endpoint = "https://rag.timetoact.at/ibm/embeddings"

    def _send_request(self, texts: list[str]) -> dict:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": self.model_id
        }
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Sends a list of documents and expects the response to contain a list of embedding vectors.
        response_data = self._send_request(texts)
        return [x['embedding'] for x in response_data["results"]]

    def embed_query(self, text: str) -> list[float]:
        # Wrap the single query text in a list as required by the API.
        response_data = self._send_request([text])
        # Return the first (and only) embedding vector.
        return response_data["results"][0]["embedding"]


if __name__ == "__main__":
    working_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent
    watson = WatsonEmbeddingProvider(path=f"{working_directory}/tokens.yaml")
    print(watson.provide().embed_documents(["Hello, world!" for x in range(1250)]))
    # openai = OpenAiEmbeddingProvider(path=f"{working_directory}/tokens.yaml")
    # print(openai.provide().embed_documents(["Hello, world!"]))
