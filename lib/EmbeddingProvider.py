import yaml
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


class EmbeddingProvider():
    def provide(self) -> Embeddings:
        pass

    # def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
    #     encoding = tiktoken.get_encoding(encoding_name)
    #     num_tokens = len(encoding.encode(string))
    #     return num_tokens


class OpenAiEmbeddingProvider(EmbeddingProvider):

    def __init__(self, path: str = "tokens.yaml", model: str = "text-embedding-3-small") -> None:
        # TODO: try different models
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
