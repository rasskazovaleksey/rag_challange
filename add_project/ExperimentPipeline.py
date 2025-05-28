import itertools
import json
import re
from collections import defaultdict
from os import walk

from langchain_community.document_loaders import PyPDFLoader

from lib.Agent import OpenAIAgent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import OpenAiEmbeddingProvider
from lib.questions import QuestionExtractor


class ExperimentPipeline:
    def __init__(self,
                 name,
                 llm,
                 repo,
                 questions_path="data/r2.0/questions.json",
                 subset_path="data/r2.0/subset.json",
                 ):
        self.extractor = QuestionExtractor()
        self.questions = self.read_questions(questions_path)
        self.subset = self.read_subset(subset_path)
        self.llm = llm
        self.repo = repo
        self.name = name

    @staticmethod
    def read_questions(path):
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def read_subset(path):
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def read_synonyms():
        filenames = next(walk("./data/r2.0/synonyms"), (None, None, []))[2]  # [] if no file

        result = []
        for f in filenames:
            with open(f"./data/r2.0/synonyms/{f}", 'r') as file:
                f = file.read()
                text = f.replace("\\n", "")
                text = text.split("Note")[0]
                pattern = r'"Here are (?:five|5) (?:synonymous(?:/similar)?|synonymic) expressions for(?: the term)? (?:\\"|")([^"]+)(?:\\"|") in JSON format:'
                text = re.sub(pattern, '', text)
                pattern = r'"Here are (?:five|5) synonymous(?:/similar)? expressions for(?: the term)? \\"[^"]+\\":'
                text = re.sub(pattern, '', text)
                pattern = r'"Here are five synonymous expressions for the term "total assets":'
                text = re.sub(pattern, '', text)
                text = text.replace("\"Here are 5 synonimatic/similar expressions for the term \\\"total assets\\\":",
                                    '')
                text = text.replace("\\\"", "\"")
                text = text.replace("]\"}", "]}")
                text = text.replace(": \" [", ": [")
                text = text.replace(": \"[", ": [")
                if text.endswith("]"):
                    text += "}"
                try:
                    j = json.loads(text)
                    result.append(j)
                except Exception as e:
                    raise e
        return result

    def extract(self, question):
        extract = self.extractor.extract(question.get("text"))
        if extract['metric'] is None:
            extract['metric'] = extract['original_question']
        extract['type'] = question.get("kind")
        extract['sha1'] = list(filter(lambda x: x["company_name"] in extract['companies'], self.subset))
        if len(extract['companies']) == 1:
            extract['sha1'] = extract['sha1'][0]["sha1"]
        else:
            extract['sha1'] = list(map(lambda x: x["sha1"], extract['sha1']))
        if extract['metric'] is None:
            raise ValueError("Metric is None")
        if extract['companies'] is None:
            raise ValueError("Companies is None")
        if extract['sha1'] is None:
            raise ValueError("Sha1 is None")
        return extract

    def get_synonyms(self, metric):
        return self.llm.query(metric, data=[], path="./prompt/synonyms_prompt.txt",
                              system="You are language specialist. Extremely precise and accurate.")

    def create_synonyms_lookup(self):
        extracts = [self.extract(q) for q in self.questions]
        for i, e in enumerate(extracts):
            answer = self.get_synonyms(e['metric'])
            result = {
                "metric": e['metric'],
                "synonyms": answer
            }
            print(result)
            json.dump(result, open(f"data/r2.0/synonyms/{i}.json", 'w'))

    def search_database(self, extract, main=10, side=5):
        sha1 = extract['sha1']
        if len(sha1) == 1:
            sha_filter = {"sha1": sha1}
            try:
                main_results = self.repo.query(extract['original_question'], k=main,
                                               f=sha_filter)  # start with main metric from the question
            except Exception as e:
                print(f"Error {e} for {extract['original_question']}")
                main_results = []
            return main_results
        else:
            main_results = []
            for s in sha1:
                try:
                    results = self.repo.query(extract['original_question'], k=main, f={"sha1": s})
                    main_results.extend(results)
                except Exception as e:
                    print(f"Error {e} for {extract['original_question']} with sha1 {s}")
            if len(main_results) == 0:
                return []
            side_results = []
            for s in sha1:
                try:
                    results = self.repo.query(extract['metric'], k=side, f={"sha1": s})
                    side_results.extend(results)
                except Exception as e:
                    print(f"Error {e} for {extract['metric']} with sha1 {s}")
            return main_results + side_results

    def filter_candidates(self, candidates, size=8):
        pages_candidates = {}
        for doc, score in candidates:
            page = doc.metadata["page"]
            if page in pages_candidates:
                pages_candidates[page]["count"] += 1
                pages_candidates[page]["score"].append(score)
            else:
                pages_candidates[page] = {
                    "count": 1,
                    "score": [score]
                }
        pcf = pages_candidates
        for p in pcf:
            pcf[p]["score"] = sum(pcf[p]["score"]) / pcf[p]["count"]

        pcf = sorted(
            pages_candidates.items(),
            key=lambda x: (-x[1]["count"], x[1]["score"])
        )
        return pcf[0:size]

    @staticmethod
    def merge_data(listed_data: list) -> list:
        merged_dict = defaultdict(lambda: {'count': 0, 'score': 0.0})

        for page, data in listed_data:
            merged_dict[page]['count'] += data['count']
            merged_dict[page]['score'] += data['score'] * data['count']

        for page in merged_dict:
            merged_dict[page]['score'] /= merged_dict[page]['count']

        merged_list = sorted(
            merged_dict.items(),
            key=lambda x: (-x[1]["count"], x[1]["score"])
        )
        return merged_list

    def read_pdf(self, sha1, candidates):
        document_loader = PyPDFLoader(f"./data/r2.0/pdfs/{sha1}.pdf")
        doc = document_loader.load()
        pages_number = [p for p, _ in candidates]
        pages = [p for p in doc if p.metadata["page"] in pages_number]
        for p in pages:
            p.metadata["id"] = p.metadata["page"]
            p.metadata["sha1"] = sha1
            assert sha1 in p.metadata["source"], f"Source {p.metadata['source']} does not contain {sha1}"

        rag = [(p, 0.0) for p in pages]
        return rag

    def run(self, data: dict):
        try:
            extract = self.extract(data)
        except Exception as e:
            return {
                "question": data.get("text"),
                "sha1": "N/A",
                "answer": "N/A",
            }

        search = self.search_database(extract, main=10, side=5)
        candidates = self.filter_candidates(search, size=10)

        documents = []
        if not isinstance(extract['sha1'], list):
            documents = self.read_pdf(extract['sha1'], candidates)
        else:
            for sha1 in extract['sha1']:
                rag = self.read_pdf(sha1, candidates)
                documents.append(rag)
            documents = list(itertools.chain.from_iterable(documents))

        answer = self.llm.query(
            text=extract['original_question'],
            data=documents,
            path=f"./prompt/{extract['type']}_prompt.txt",
            system="You are competent financial analytic.")

        referenced_answer = {
            "question": extract['original_question'],
            "sha1": extract['sha1'],
            "answer": answer
        }
        print(f"Found answer: {referenced_answer['answer']}")

        return referenced_answer


if __name__ == "__main__":
    ExperimentPipeline(
        name="openai_small_1000_100_filtered_v1",
        llm=OpenAIAgent(),
        repo=DataRepository(
            embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
            db_path="./data/db/open_ai_small_1000_100_filtered",
            path="./data/r2.0/pdfs",
            name="open_ai_small_1000_100_filtered",
            chunk_size=1000,
            chunk_overlap=100),
    ).run()
