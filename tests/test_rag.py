import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, project_root)

from pathlib import Path

from lib.Agent import OpenAIAgent, Agent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider


def run_test(
        agent: Agent,
        repo: DataRepository,
        working_directory: str,
        path: str = './data/r2.0-test/questions_with_answer.json',
) -> None:
    with open(path, 'r') as file:
        questions = json.load(file)

    resultHolder = []
    for i, item in enumerate(questions):
        question = item.get("text")
        expected_answer = item.get("answer")
        question_type = item.get("kind")

        if question_type == 'name':
            question_type = 'names' #one question hame type name, while other nameS

        data = repo.query(question)
        actual = agent.query(question, data, path=f'{working_directory}/prompt/{question_type}_prompt.txt')
        
        
        if ' (' in actual:
            actual, chunk_id = actual.split(' (ID: ')
            chunk_id = chunk_id.rstrip(')')
        else:
            chunk_id = 'N/A'
            
        if expected_answer == actual:
            print(f"✅ Test passed: '{question}'")
            print(f"   Expected answer: {expected_answer}")
            print(f"   Real answer: {actual}")
            print(f"   ID: {chunk_id}")
            resultHolder.append({
                "isPass": True,
                "question": question,
                "expected_answer": expected_answer,
            })
        else:
            print(f"❌ Test failed: '{question}'")
            print(f"   Expected answer: {expected_answer}")
            print(f"   Real answer: {actual}")
            resultHolder.append({
                "isPass": False,
                "question": question,
                "expected_answer": expected_answer,
            })

    passed = len([x for x in resultHolder if x['isPass']])
    failed = len([x for x in resultHolder if not x['isPass']])
    print(f"Total: {len(resultHolder)}, Passed: {passed}, Failed: {failed}")


if __name__ == "__main__":
    working_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent

    agent = OpenAIAgent(path=f"{working_directory}/tokens.yaml")

    repo = DataRepository(
        embedding=OpenAiEmbeddingProvider(f"{working_directory}/tokens.yaml"),
        db_path=f"{working_directory}/data/db/open_ai_small"
    )

    run_test(
        path=f"{working_directory}/data/r2.0-test/questions_with_answer.json",
        agent=agent,
        repo=repo,
        working_directory=working_directory
    )
