import json
import os
import sys

from langchain_community.document_loaders import PyPDFLoader

from lib.questions import QuestionExtractor

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

        extractor = QuestionExtractor()
        extract = extractor.extract(question)
        close_metrics = extractor.get_synonyms(extract["metric"])
        companiy = extract["companies"][0]
        with open(f"{working_directory}/data/r2.0-test/subset.json", 'r') as file:
            subset = json.load(file)
        filered = list(filter(lambda x: x["company_name"] == companiy, subset))[0]
        main_metric = extract["metric"]
        file_filter = {"source": f"./data/r2.0-test/pdfs/{filered["sha1"]}.pdf"}
        main_results = repo.query(main_metric, k=10, f=file_filter)  # start with main metric from the question

        main_metric = extract["metric"]
        smaller_results = []  # start with main metric from the question
        for m in close_metrics:
            smaller_results += repo.query(m, k=5)  # find similar metrics
        search_results = main_results + smaller_results
        pages_candidates = {}
        for doc, score in search_results:
            page = doc.metadata["page"]
            if page in pages_candidates:
                pages_candidates[page]["count"] += 1
                pages_candidates[page]["score"].append(score)
            else:
                pages_candidates[page] = {
                    "count": 1,
                    "score": [score]
                }
        pages_candidates_filtered = pages_candidates
        for p in pages_candidates_filtered:
            pages_candidates_filtered[p]["score"] = sum(pages_candidates_filtered[p]["score"]) / \
                                                    pages_candidates_filtered[p]["count"]

        pages_candidates_filtered = sorted(
            pages_candidates.items(),
            key=lambda x: (-x[1]["count"], x[1]["score"])
        )

        document_loader = PyPDFLoader(f"{working_directory}/{file_filter["source"].replace('./', '')}")
        doc = document_loader.load()

        pages = [p for p in doc if p.metadata["page"] in [p for p, _ in pages_candidates_filtered[0:8]]]
        for p in pages:
            p.metadata["id"] = p.metadata["page"]

        rag = [(p, 0.0) for p in pages]

        actual = agent.query(question, rag, path=f'{working_directory}/prompt/{question_type}_prompt.txt')

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
        db_path=f"{working_directory}/data/db/open_ai_small_50_10"
    )

    run_test(
        path=f"{working_directory}/data/r2.0-test/questions_with_answer.json",
        agent=agent,
        repo=repo,
        working_directory=working_directory
    )
