from dataclasses import dataclass


@dataclass
class RabbitMQConfig:
    host: str
    port: int
    username: str
    password: str


class ConfigLoader:
    @staticmethod
    def load() -> RabbitMQConfig:
        return RabbitMQConfig(host='localhost',
                              port=5672,
                              username='guest',
                              password='guest',
                              )
