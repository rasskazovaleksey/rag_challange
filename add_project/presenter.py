import sys
import json
from rabbit_core.json_sender import JsonSender  
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory


class Presenter:
    def __init__(self, json_sender: JsonSender):
        self.json_sender = json_sender

    def start(self):
        print("Presenter is listening for JSON input (Ctrl+C to stop):")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if isinstance(data, list):
                    for i in data:
                        self.json_sender.send_json(i)
                else:
                    self.json_sender.send_json(data)

            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON: {e}")


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    sender = JsonSender(connection)

    presenter = Presenter(json_sender=sender)
    presenter.start()
