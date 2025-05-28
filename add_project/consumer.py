import json
from rabbit_core.json_receiver import JsonReceiver
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory
from ExperimentPipeline import ExperimentPipeline
from lib.Agent import OpenAIAgent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


class JsonProcessor:
    def __init__(self, connection, pipeline:ExperimentPipeline):
        self.receiver = JsonReceiver(connection)
        self.pipeline = pipeline
        self.receiver.register_callback(self._on_message)

    def start(self):
        print("[JsonProcessor] Waiting for JSON messages...")
        try:
            self.receiver.start_consuming()
        except KeyboardInterrupt:
            print("\n[JsonProcessor] Stopped by user.")

    def _on_message(self, data: dict):
        print("[JsonProcessor] Message received. Processing...")
        self.handle_json(data)

    def handle_json(self, data: dict):
        self.pipeline.run()
        #print(json.dumps(data, indent=2))
        


if __name__ == '__main__':
    config = ConfigLoader().load()
    factory = ConnectionFactory(config)
    connection = factory.create_connection()
    pipeline = ExperimentPipeline(
            name="openai_small_1000_100_filtered_v1",
            llm=OpenAIAgent(),
            repo=DataRepository(
                embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
                db_path="./data/db/open_ai_small_1000_100_filtered",
                path="./data/r2.0/pdfs",
                name="open_ai_small_1000_100_filtered",
                chunk_size=1000,
                chunk_overlap=100),
        )
    processor = JsonProcessor(connection, pipeline)
    processor.start()
