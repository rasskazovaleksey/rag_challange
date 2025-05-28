import pika
from .config_loader import RabbitMQConfig

class ConnectionFactory:
    def __init__(self, config: RabbitMQConfig):
        self.config = config

    def create_connection(self):
        credentials = pika.PlainCredentials(self.config.username, self.config.password)
        parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            credentials=credentials
        )
        return pika.BlockingConnection(parameters)
