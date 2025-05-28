from os import listdir
from os.path import isfile, join, basename
from rabbit_core.pdf_sender import PdfSender
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory

class PdfProducerService:

    def __init__(self, rabbit_sender: PdfSender) -> None:
        self.sender = rabbit_sender

    def send_files(self, path: str = "pdfs/"):
        files = [f"{path}/{f}" for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            print(f"Progress: {files.index(file)}/{len(files) - 1}")
            with open(file, "rb") as f:
                file_bytes = f.read()
            file_name = basename(file)
            self.sender.send_pdf(file_name, file_bytes)
if __name__ == '__main__':

    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    sender = PdfSender(connection)
    

    producer = PdfProducerService(
            rabbit_sender=sender
            )

        # producer = PdfProducerService(
    #         embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
    #         rabbit_sender=sender,
    #         db_path="./data/db/open_ai_small_1000_100_filtered",
    #         path="./data/r2.0/pdfs",
    #         name="open_ai_small_1000_100_filtered",
    #         chunk_size=1000,
    #         chunk_overlap=100
    #         )
    producer.send_files()