import yaml
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
