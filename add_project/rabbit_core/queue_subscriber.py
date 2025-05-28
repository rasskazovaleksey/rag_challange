from .connection_factory import ConnectionFactory
from pika.exceptions import AMQPConnectionError
from time import sleep


class QueueSubscriber:
    def __init__(self, connection: ConnectionFactory, queue_name: str):
        self.channel = connection.channel()
        self.channel.queue_declare(queue=queue_name, durable=True)

    def subscribe(self, callback):
        try:
            self.channel.basic_qos(prefetch_count=30)
            self.channel.basic_consume(
                queue=self.queue_name, on_message_callback=callback, auto_ack=False
            )
            
            self.channel.start_consuming()

        except AMQPConnectionError:
            sleep(5)
