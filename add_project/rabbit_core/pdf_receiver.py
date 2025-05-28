import pika
import os
from .config_loader import ConfigLoader
from .connection_factory import ConnectionFactory


class PdfReceiver:
    def __init__(self, connection: pika.BlockingConnection, output_dir="output"):
        self.output_dir = output_dir
        #os.makedirs(output_dir, exist_ok=True)
        self.connection = connection
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='pdf_exchange', exchange_type='direct')
        self.channel.queue_declare(queue='pdf_queue')
        self.channel.queue_bind(exchange='pdf_exchange', queue='pdf_queue', routing_key='pdf')
        self.buffers = {}
        self.on_pdf_received = None  # Колбэк

    def register_callback(self, callback):
        self.on_pdf_received = callback

    def _callback(self, ch, method, properties, body):
        file_name = properties.headers['file_name']
        chunk_number = properties.headers['chunk_number']
        is_last_chunk = properties.headers['is_last_chunk']

        if file_name not in self.buffers:
            self.buffers[file_name] = {}

        if body:
            self.buffers[file_name][chunk_number] = body

        if is_last_chunk:
            print(f"Reassembling file {file_name} from {len(self.buffers[file_name])} chunks...")
            pdf_bytes = b''.join(self.buffers[file_name][i] for i in sorted(self.buffers[file_name].keys()))
            del self.buffers[file_name]

            if self.on_pdf_received:
                self.on_pdf_received(pdf_bytes, file_name)

    def start_consuming(self):
        self.channel.basic_consume(
            queue='pdf_queue',
            on_message_callback=self._callback,
            auto_ack=True
        )
        print("Waiting for messages...")
        self.channel.start_consuming()



if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    receiver = PdfReceiver(connection)
    receiver.start_consuming()
