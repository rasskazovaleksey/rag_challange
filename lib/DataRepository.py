from os import listdir
from os.path import isfile, join
from typing import Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.EmbeddingProvider import EmbeddingProvider


class DataRepository:

    def __init__(self, embedding: EmbeddingProvider, name: str = "open_ai_small", path: str = "./data/r2.0-test/pdfs",
                 db_path: str = "./data/db/", chunk_size: int = 1_000, chunk_overlap: int = 100) -> None:
        self.embedding = embedding
        self.path = path
        self.db = Chroma(collection_name=name, persist_directory=db_path, embedding_function=self.embedding.provide())
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    @staticmethod
    def __load_documents(path: str) -> list[Document]:
        document_loader = PyPDFLoader(path)
        return document_loader.load()

    def __split(self, documents: list[Document]) -> list[Document]:
        return self.__text_splitter.split_documents(documents)

    @staticmethod
    def __append_chunk_ids(splits: list[Document]) -> list[Document]:
        last_page_id = None
        current_chunk_index = 0

        for chunk in splits:
            source = chunk.metadata.get("source").split('/')[-1].replace(".pdf", "")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id
        return splits

    @staticmethod
    def __filter(doc: Document) -> Document:
        doc.page_content = doc.page_content.replace("\n", " ")
        # TODO: might be a good idea to filter out some of the text as doc.page_content = "my content"
        return doc

    def __create(self, documents: list[Document]) -> None:
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = []
        for chunk in documents:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            max_batch_size = 5461
            if len(new_chunks) > max_batch_size:
                print("Adding documents in batches...")
                for i in range(0, len(new_chunks), max_batch_size):
                    self.db.add_documents(new_chunks[i:i + max_batch_size], ids=new_chunk_ids[i:i + max_batch_size])
            else:
                self.db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    def save_by_file(self, path: str = "./data/r2.0-test/pdfs"):
        files = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            print(f"Progress: {files.index(file)}/{len(files) - 1}")
            print(f"Loading documents from {file}...")
            docs = self.__load_documents(file)
            print("Splitting documents...")
            splits = self.__split(docs)
            print("Appending chunk ids...")
            splits = self.__append_chunk_ids(splits)
            print("Filtering documents...")
            splits = [self.__filter(doc) for doc in splits]
            print(f"Number of chunks: {len(splits)}")
            self.__create(splits)

    def query(self, text: str, k=5, f: dict[str, str] = None) -> list[Tuple[Document, float]]:
        results = self.db.similarity_search_with_score(text, k=k, filter=f)
        return results
