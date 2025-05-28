import json
import os

from ExperimentPipeline import ExperimentPipeline
from lib.Agent import OpenAIAgent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider
from rabbit_core.config_loader import ConfigLoader
from rabbit_core.connection_factory import ConnectionFactory
from rabbit_core.json_receiver import JsonReceiver


class JsonProcessor:
    def __init__(self, connection, pipeline: ExperimentPipeline):
        self.receiver = JsonReceiver(connection)
        self.pipeline = pipeline
        self.receiver.register_callback(self._on_message)
        self.answer_file_path = "answer.json"

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
        print(f"Processing data: {json.dumps(data, indent=2)}")

        self.pipeline.run(data)


        # answers = []
        # if os.path.exists(self.answer_file_path):
        #     try:
        #         with open(self.answer_file_path, "r") as f:
        #             content = f.read()
        #             if content:  # Check if file is not empty
        #                 answers = json.loads(content)
        #             if not isinstance(answers, list):  # Ensure answers is a list
        #                 print(
        #                     f"[JsonProcessor] Warning: '{self.answer_file_path}' did not contain a JSON list. Initializing as empty list.")
        #                 answers = []
        #     except json.JSONDecodeError:
        #         print(
        #             f"[JsonProcessor] Warning: Could not decode JSON from '{self.answer_file_path}'. Initializing as empty list.")
        #         answers = []
        #     except Exception as e:
        #         print(f"[JsonProcessor] Error reading '{self.answer_file_path}': {e}. Initializing as empty list.")
        #         answers = []
        #
        # question_text = data.get("text")
        # if not question_text:
        #     print("[JsonProcessor] Error: 'text' field missing in input data.")
        #     return
        #
        # filtered = [a for a in answers if a.get("question") == question_text]
        #
        # if filtered:
        #     print(f"[JsonProcessor] Answer already exists for question: {question_text}")
        #     return
        # else:
        #     print(f"[JsonProcessor] No existing answer found for: {question_text}. Running pipeline.")
        #     new_answer_data = self.pipeline.run(data)
        #     answers.append(new_answer_data)
        #     try:
        #         with open(self.answer_file_path, "w") as f:
        #             json.dump(answers, f, indent=4)
        #         print(f"[JsonProcessor] Successfully updated '{self.answer_file_path}'")
        #     except IOError as e:
        #         print(f"[JsonProcessor] Error writing to file '{self.answer_file_path}': {e}")


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