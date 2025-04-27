import pika
import os

class PdfSender:
    def __init__(self, config):
        self.config = config
        credentials = pika.PlainCredentials(config['username'], config['password'])
        parameters = pika.ConnectionParameters(
            host=config['host'],
            port=config['port'],
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=config['exchange'], exchange_type='direct')

    def send_pdf(self, file_path, chunk_size=1024 * 50):  # 50 KB per chunk
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            chunk_number = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                properties = pika.BasicProperties(
                    headers={
                        'file_name': file_name,
                        'chunk_number': chunk_number,
                        'is_last_chunk': False
                    }
                )
                self.channel.basic_publish(
                    exchange=self.config['exchange'],
                    routing_key=self.config['routing_key'],
                    body=chunk,
                    properties=properties
                )
                chunk_number += 1

            self.channel.basic_publish(
                exchange=self.config['exchange'],
                routing_key=self.config['routing_key'],
                body=b'',
                properties=pika.BasicProperties(
                    headers={
                        'file_name': file_name,
                        'chunk_number': chunk_number,
                        'is_last_chunk': True
                    }
                )
            )
        print(f"File {file_name} has been sent in chunks.")

    def close(self):
        self.connection.close()
