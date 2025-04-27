import yaml

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_rabbitmq_config(self):
        return self.config.get("rabbitmq", {})
