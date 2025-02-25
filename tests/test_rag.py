import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, project_root)

from pathlib import Path

from lib.Agent import OpenAIAgent, Agent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


def run_test(
             path:str = './data/r2.0-test/questions_with_answer.json', 
             working_directory:str = None, 
             agent:Agent = None, 
             repo:DataRepository = None,
             asrt = True
             ) -> None:

    with open(path, 'r') as file:
        questions = json.load(file)

    for item in questions:
        question = item.get("text")
        expected_answer = item.get("answer")

        data = repo.query(question)
        resp = agent.query(question, data, f"{working_directory}/prompt/generic_prompt.txt")

        if asrt:
            assert expected_answer == resp, f"Error in question: '{question}'\nAgent: {agent}\nExpected answer: {expected_answer}\nReal answer: {resp}"
            
            print(f"Test passed: '{question}'")

        else:
            if expected_answer == resp:
                print(f"✅ Test passed: '{question}'")
            else:
                print(f"❌ Test failed: '{question}'")
                print(f"   Expected answer: {expected_answer}")
                print(f"   Real answer: {resp}")


if __name__ == "__main__":
    working_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    agent = OpenAIAgent(path=f"{working_directory}/tokens.yaml")

    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(f"{working_directory}/tokens.yaml"),
        db_path=f"{working_directory}/data/db/open_ai_small"
    )

    run_test(
             path=f"{working_directory}//data/r2.0-test/questions_with_answer.json",
             working_directory=working_directory, 
             agent=agent, 
              )




