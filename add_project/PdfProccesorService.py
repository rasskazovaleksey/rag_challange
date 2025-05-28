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
from rabbit_core.pdf_receiver import PdfReceiver
from rabbit_core.pdf_sender import PdfSender
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory
from lib.EmbeddingProvider import OpenAiEmbeddingProvider
from PdfRepo import PdfRepo

class PdfReceiverService:

    def __init__(
        self,
        #embedding: EmbeddingProvider,
        rabbit_receiver: PdfReceiver,
        repo: PdfRepo,  # <--- добавляем сюда
        #name: str = "open_ai_small",
        #path: str = "./data/r2.0-test/pdfs",
        #db_path: str = "./data/db/",
        chunk_size: int = 1_000,
        chunk_overlap: int = 100
    ) -> None:
        #self.embedding = embedding
        #self.path = path
        #self.name = name
        self.receiver = rabbit_receiver
        self.repo = repo 
        #self.db = Chroma(collection_name=name, persist_directory=db_path, embedding_function=self.embedding.provide())
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.STOPWORDS = set(stopwords.words('english'))
    
    @staticmethod
    def __load_documents(pdf_bytes: bytes) -> list[Document]: #well, PyPDFLoader can work only with path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        finally:
            os.remove(tmp_path)  # Temp file del

        return documents


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
            chunk.metadata["sha1"] = source
        return splits

    def __filter(self, doc: Document) -> Document:
        doc.page_content = doc.page_content.replace("\n", " ")
        doc.page_content = self.__clean_text(doc.page_content)
        # dirty trick for Watson
        doc.page_content = doc.page_content.replace("ŷǉĩžƌǁăƌěͳůžžŭŝŷőɛƚăƚğŵğŷƚɛŝŷžƶƌěŝɛđƶɛɛŝžŷăƌğďăɛğěžŷƚśğğǆɖğđƚăɵžŷɛğɛɵŵăƚğɛăŷěɖƌžũğđɵžŷɛžĩŵăŷăőğŵğŷƚăɛžĩƚžěăǉăŷě ăƌğɛƶďũğđƚƚžǀăƌŝžƶɛăɛɛƶŵɖɵžŷɛƌŝɛŭɛƶŷđğƌƚăŝŷɵğɛăŷěžƚśğƌĩăđƚžƌɛƚśăƚăƌğěŝĸđƶůƚƚžɖƌğěŝđƚǁśŝđśđžƶůěđăƶɛğăđƚƶăůƌğɛƶůƚɛƚžěŝīğƌ ŵăƚğƌŝăůůǉĩƌžŵƚśžɛğğǆɖƌğɛɛğěžƌŝŵɖůŝğěŝŷƚśğĩžƌǁăƌěͳůžžŭŝŷőɛƚăƚğŵğŷƚɛdśğɛğɛƚăƚğŵğŷƚɛăƌğŷžƚőƶăƌăŷƚğğɛžĩĩƶƚƶƌğɖğƌĩžƌŵăŷđğăŷě ƚśğƌğĩžƌğƶŷěƶğƌğůŝăŷđğɛśžƶůěŷžƚďğɖůăđğěƶɖžŷƚśğŵtğƌğĩğƌăůůžĩǉžƶƚžžƶƌϯϭϯϯŷŷƶăůzğɖžƌƚžŷžƌŵϭϭͳăŷěžƶƌžƚśğƌįůŝŷőɛǁŝƚś ƚśğ ĩžƌăŵžƌğěğƚăŝůğěěŝɛđƶɛɛŝžŷžĩƚśğƌŝɛŭɛƚśăƚđžƶůěŝŵɖăđƚƚśğĩƶƚƶƌğžɖğƌăɵŷőƌğɛƶůƚɛăŷěįŷăŷđŝăůđžŷěŝɵžŷžĩzƶŵďůğkŷŷđtğ ěŝɛđůăŝŵăŷǉŝŷƚğŷɵžŷɛžƌžďůŝőăɵžŷɛƚžƶɖěăƚğžƌƌğǀŝɛğăŷǉĩžƌǁăƌěͳůžžŭŝŷőɛƚăƚğŵğŷƚɛğǆđğɖƚăɛƌğƌƶŝƌğěďǉůăǁ", "")
        return doc

    def __clean_text(self, text):
        # Original Text
        # Example: "This is a Testing @username https://example.com <p>Paragraphs!</p> #happy :)"

        text = text.lower()  # Convert all characters in text to lowercase
        # Example after this step: "i won't go there! this is a testing @username https://example.com <p>paragraphs!</p> #happy :)"

        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        # Example after this step: "i won't go there! this is a testing @username  <p>paragraphs!</p> #happy :)"

        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        # Example after this step: "i won't go there! this is a testing @username  paragraphs! #happy :)"

        text = re.sub(r'@\w+', '', text)  # Remove mentions
        # Example after this step: "i won't go there! this is a testing   paragraphs! #happy :)"

        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        # Example after this step: "i won't go there! this is a testing   paragraphs!  :)"

        # Translate emoticons to their word equivalents
        emoticons = {':)': 'smile', ':-)': 'smile', ':(': 'sad', ':-(': 'sad'}
        words = text.split()
        words = [emoticons.get(word, word) for word in words]
        text = " ".join(words)
        # Example after this step: "i won't go there! this is a testing paragraphs! smile"

        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
        # Example after this step: "i won't go there this is a testing paragraphs smile"

        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove standalone single alphabetical characters
        # Example after this step: "won't go there this is testing paragraphs smile"

        text = re.sub(r'\s+', ' ', text, flags=re.I)  # Substitute multiple consecutive spaces with a single space
        # Example after this step: "won't go there this is testing paragraphs smile"

        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
        # Example after this step: "won't go there testing paragraphs smile"

        # Stemming
        # stemmer = PorterStemmer()
        # text = ' '.join(stemmer.stem(word) for word in text.split())
        # Example after this step: "won't go there test paragraph smile"

        # Lemmatization. (flies --> fly, went --> go)
        # lemmatizer = WordNetLemmatizer()
        # text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

        return text

    def save_by_file(self, path: str = "./data/r2.0-test/pdfs"):
        files = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            print(f"Progress: {files.index(file)}/{len(files) - 1}")
            docs = self.__load_documents(file)
            splits = self.__split(docs)
            splits = self.__append_chunk_ids(splits)
            splits = [self.__filter(doc) for doc in splits]
    
    def handle_received_pdf(self, pdf_bytes: bytes, file_name: str):
        print(f"Processing file: {file_name}")

        docs = self.__load_documents(pdf_bytes)
        splits = self.__split(docs)
        splits = self.__append_chunk_ids(splits)
        splits = [self.__filter(doc) for doc in splits]

        self.repo.create(splits)
        print(f"Finished processing and saving: {file_name}")


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    receiver = PdfReceiver(connection)

    repo = PdfRepo(
        embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
        db_path="./data/db/open_ai_small_1000_100_filtered",
        path="./data/r2.0/pdfs",
        name="open_ai_small_1000_100_filtered",
    
    )

    service = PdfReceiverService(
        #embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
        rabbit_receiver=receiver,
        repo=repo,
        #db_path="./data/db/open_ai_small_1000_100_filtered",
        #path="./data/r2.0/pdfs",
        #name="open_ai_small_1000_100_filtered",
        chunk_size=1000,
        chunk_overlap=100
    )

    receiver.register_callback(service.handle_received_pdf)
    receiver.start_consuming()


