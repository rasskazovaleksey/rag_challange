import pika
import os
from .config_loader import ConfigLoader
from .connection_factory import ConnectionFactory

class PdfSender:
    def __init__(self, connection: pika.BlockingConnection):
        self.channel = connection.channel()
        self.channel.exchange_declare(exchange='pdf_exchange', exchange_type='direct')

    def send_pdf(self, file_name: str, file_bytes: bytes, chunk_size=1024 * 50):  # 50 KB per chunk
        chunk_number = 0
        offset = 0
        total_length = len(file_bytes)

        while offset < total_length:
            chunk = file_bytes[offset:offset + chunk_size]
            offset += chunk_size

            is_last_chunk = offset >= total_length

            properties = pika.BasicProperties(
                headers={
                    'file_name': file_name,
                    'chunk_number': chunk_number,
                    'is_last_chunk': is_last_chunk
                }
            )

            self.channel.basic_publish(
                exchange='pdf_exchange',
                routing_key='pdf',
                body=chunk,
                properties=properties
            )

            chunk_number += 1

        print(f"File {file_name} has been sent in chunks.")


    # def close(self):
    #     self.connection.close()



if __name__ == '__main__':

    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    sender = PdfSender(connection)

    sender.send_pdf(f"pdfs/0c0faea14d108e1617f2d6d2a7c1aae04eb88fe0.pdf")
    #sender.close()