import re
from os import listdir
from os.path import isfile, join
from typing import Tuple

from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.EmbeddingProvider import EmbeddingProvider

import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader


class PdfRepo:

    def __init__(self, embedding: EmbeddingProvider, name: str = "open_ai_small", path: str = "./data/r2.0-test/pdfs",
                 db_path: str = "./data/db/") -> None:
        self.embedding = embedding
        self.path = path
        self.name = name
        self.db = Chroma(collection_name=name, persist_directory=db_path, embedding_function=self.embedding.provide())
        self.STOPWORDS = set(stopwords.words('english'))
    
    def create(self, documents: list[Document]) -> None:
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
            max_batch_size = 1000
            if len(new_chunks) > max_batch_size:
                print("Adding documents in batches...")
                for i in range(0, len(new_chunks), max_batch_size):
                    self.db.add_documents(new_chunks[i:i + max_batch_size], ids=new_chunk_ids[i:i + max_batch_size])
            else:
                self.db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    def query(self, text: str, k=5, f: dict[str, str] = None) -> list[Tuple[Document, float]]:
        results = self.db.similarity_search_with_score(text, k=k, filter=f)
        return results
    
    