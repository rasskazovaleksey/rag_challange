import pika
import os
from config_loader import ConfigLoader
from connection_factory import ConnectionFactory

class PdfReceiver:
    def __init__(self, connection: pika.BlockingConnection, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.connection = connection
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='pdf_exchange', exchange_type='direct')
        self.channel.queue_declare(queue='pdf_queue')
        self.channel.queue_bind(exchange='pdf_exchange', queue='pdf_queue', routing_key='pdf')

        self.buffers = {}

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
            full_path = os.path.join(self.output_dir, file_name)
            with open(full_path, 'wb') as f:
                for i in range(len(self.buffers[file_name])):
                    f.write(self.buffers[file_name][i])
            print(f"File {file_name} has been reconstructed and saved to {full_path}")
            del self.buffers[file_name]

    def start_consuming(self):
        self.channel.basic_consume(
            queue='pdf_queue',
            on_message_callback=self._callback,
            auto_ack=True
        )
        print("Waiting for messages...")
        self.channel.start_consuming()

    def close(self):
        self.connection.close()


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    receiver = PdfReceiver(connection)
    receiver.start_consuming()
