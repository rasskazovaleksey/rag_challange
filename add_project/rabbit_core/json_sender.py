import pika
import json
from .config_loader import ConfigLoader
from .connection_factory import ConnectionFactory

import asyncore


class JsonSender:
    def __init__(self, connection: pika.BlockingConnection):
        self.exchange = 'json_exchange'
        self.channel = connection.channel()
        self.channel.exchange_declare(exchange=self.exchange, exchange_type='direct')

    def send_json(self, data: dict, chunk_size=1024 * 50):
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        chunk_number = 0
        offset = 0
        total_length = len(json_bytes)

        while offset < total_length:
            chunk = json_bytes[offset:offset + chunk_size]
            offset += chunk_size

            is_last_chunk = offset >= total_length

            properties = pika.BasicProperties(
                headers={
                    'chunk_number': chunk_number,
                    'is_last_chunk': is_last_chunk
                }
            )

            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key='json',
                body=chunk,
                properties=properties
            )

            chunk_number += 1

        print(f"JSON has been sent in chunks.")


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    sender = JsonSender(connection)

    sample_json = {
        "message": "This is a test JSON message.",
        "data": [1, 2, 3, 4, 5],
        "meta": {"author": "Tester", "purpose": "Chunked transmission test"}
    }

    sender.send_json(data=sample_json)
