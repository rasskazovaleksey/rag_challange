import os
from pathlib import Path
from typing import Tuple

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.EmbeddingProvider import EmbeddingProvider


class DataRepository:

    def __init__(self, embedding: EmbeddingProvider, name: str = "open_ai_small", db_path: str = "./data/db/") -> None:
        self.embedding = embedding
        self.db = Chroma(collection_name=name, persist_directory=db_path, embedding_function=self.embedding.provide())

    __text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    def _load_documents(self, path: str = "./data/r2.0-test/pdfs") -> list[Document]:
        document_loader = PyPDFDirectoryLoader(path)
        return document_loader.load()

    def _split(self, documents: list[Document]) -> list[Document]:
        return self.__text_splitter.split_documents(documents)

    def _append_chunk_ids(self, splits: list[Document]) -> list[Document]:
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

    def _filter(self, doc: Document) -> Document:
        doc.page_content = doc.page_content.replace("\n", " ")
        # TODO: might be a good idea to filter out some of the text as doc.page_content = "my content"
        return doc

    def _prepare_chunks(self, path: str = "./data/r2.0-test/pdfs") -> list[Document]:
        print("Loading documents...")
        docs = self._load_documents(path)
        print("Splitting documents...")
        splits = self._split(docs)
        print("Appending chunk ids...")
        splits = self._append_chunk_ids(splits)
        print("Filtering documents...")
        splits = [self._filter(doc) for doc in splits]
        print(f"Number of chunks: {len(splits)}")
        return splits

    def _create(self, documents: list[Document]) -> None:
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
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    def save(self):
        documents = self._prepare_chunks()
        self._create(documents)

    def query(self, text: str, k=5) -> list[Tuple[Document, float]]:
        results = self.db.similarity_search_with_score(text, k=k)
        return results
