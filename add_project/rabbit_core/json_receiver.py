import pika
import json
from .config_loader import ConfigLoader
from .connection_factory import ConnectionFactory


class JsonReceiver:
    def __init__(self, connection: pika.BlockingConnection):
        self.connection = connection
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='json_exchange', exchange_type='direct')
        self.channel.queue_declare(queue='json_queue')
        self.channel.queue_bind(exchange='json_exchange', queue='json_queue', routing_key='json')
        self._json_buffer = {}
        self.on_json_received = None

    def register_callback(self, callback):
        self.on_json_received = callback

    def _callback(self, ch, method, properties, body):
        try:
            json_data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[JSON decode error] {e}")
            return

        if self.on_json_received:
            self.on_json_received(json_data)

    def start_consuming(self):
        self.channel.basic_consume(
            queue='json_queue',
            on_message_callback=self._callback,
            auto_ack=True
        )
        print("Waiting for JSON messages...")
        self.channel.start_consuming()


def handle_json(data: dict):
    print(f"\nReceived JSON:")
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()

    receiver = JsonReceiver(connection)
    receiver.register_callback(handle_json)
    receiver.start_consuming()
