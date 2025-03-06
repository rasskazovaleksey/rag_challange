import json
import re
from collections import defaultdict
from os import walk

from langchain_community.document_loaders import PyPDFLoader

from lib.Agent import IBMWatsonAgent, OpenAIAgent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import WatsonEmbeddingProvider, OpenAiEmbeddingProvider
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

    def search_database(self, synonyms, extract, main=10, side=5):
        sha1 = extract['sha1']
        assert synonyms['metric'] == extract['metric']
        sha_filter = {"sha1": sha1}
        try:
            main_results = self.repo.query(synonyms['metric'], k=main,
                                           f=sha_filter)  # start with main metric from the question
        except Exception as e:
            print(f"Error {e} for {synonyms['metric']}")
            main_results = []
        smaller_results = []  # start with main metric from the question
        for m in synonyms['synonyms']:
            try:
                smaller_results += self.repo.query(m['text'], k=side, f=sha_filter)  # find similar metrics
            except Exception as e:
                print(f"Error {e} for {m}")
        return main_results + smaller_results

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

    def run(self):
        print(f"Starting the pipeline ${self.name} with repo {self.repo.name} and llm {self.llm.model}")
        extracts = [self.extract(q) for q in self.questions]
        synonyms_lookup = self.read_synonyms()
        answers = []
        for i, e in enumerate(extracts):
            print(f"Processing {i}/{len(extracts) - 1} with sha1 {e['sha1']}")
            if len(e['companies']) == 1 or len(e['companies']) == 4 or len(e['companies']) == 5 or len(
                    e['companies']) == 6:
                pass
            else:
                raise ValueError(f"Companies is {len(e['companies'])} for {e}")

            if len(e['companies']) == 1:
                synonyms = list(filter(lambda x: x['metric'] == e['metric'], synonyms_lookup))[0]
                all_candidates = self.search_database(synonyms, e, main=10, side=5)
                candidates = self.filter_candidates(all_candidates, size=8)
                documents = self.read_pdf(e['sha1'], candidates)

                question_type = e['type']
                if question_type == 'name':
                    question_type = 'names'
                answer = self.llm.query(e['original_question'], data=documents,
                                        path=f"./prompt/{question_type}_prompt.txt")

                answers.append({
                    "extract": e,
                    "answer": answer
                })
            else:
                print(f"Comparison problem found for {e['companies']}")
                holder = {}
                for c in e['companies']:
                    if c == 'Inc.':
                        print("--- Inc. found, skipping its and error in the code. ---")
                        continue
                    print(f"Processing {c}")
                    l = list(filter(lambda x: c in x['company_name'], self.subset))
                    sha1 = l[0]['sha1']
                    copy_e = e.copy()
                    copy_e['sha1'] = sha1
                    synonyms = list(filter(lambda x: x['metric'] == e['metric'], synonyms_lookup))[0]
                    all_candidates = self.search_database(synonyms, copy_e, main=10, side=5)
                    candidates = self.filter_candidates(all_candidates, size=8)
                    documents = self.read_pdf(sha1, candidates)
                    question = f"What was the {e['metric']} of {c} in the period? If data is not available, return 'N/A'."
                    answer = self.llm.query(question, data=documents, path=f"./prompt/number_prompt.txt",
                                            system="You are a data extraction engine with financial knowlage.")
                    holder[c] = answer

                if e['comparison'] is None:
                    raise ValueError("Comparison is None")
                holder['comparison'] = e['comparison']

                answers.append({
                    "extract": e,
                    "holder": holder
                })
        json.dump(answers, open(f"{self.name}.json", 'w'))


if __name__ == "__main__":
    ExperimentPipeline(
        name="watson_small_llama_405b_v1",
        llm=IBMWatsonAgent(model="meta-llama/llama-3-405b-instruct"),
        repo=DataRepository(
            embedding=WatsonEmbeddingProvider(),
            db_path="./data/db/watson_ai_large_100_10_filtered",
            path="./data/r2.0/pdfs",
            name="watson_ai_large_100_10_filtered",
            chunk_size=100,
            chunk_overlap=10),
    ).run()

    # ExperimentPipeline(
    #     name="watson_large_llama_405b_v1",
    #     llm=IBMWatsonAgent(model="meta-llama/llama-3-405b-instruct"),
    #     repo=DataRepository(
    #         embedding=WatsonEmbeddingProvider(),
    #         db_path="./data/db/watson_ai_large_1000_100_filtered",
    #         path="./data/r2.0/pdfs",
    #         name="watson_ai_large_1000_100_filtered",
    #         chunk_size=1000,
    #         chunk_overlap=100),
    # ).run()
    #
    # ExperimentPipeline(
    #     name="openai_large_100_10_v1",
    #     llm=OpenAIAgent(),
    #     repo=DataRepository(
    #         embedding=OpenAiEmbeddingProvider(model="text-embedding-3-large"),
    #         db_path="./data/db/open_ai_large_100_10",
    #         path="./data/r2.0/pdfs",
    #         name="open_ai_large_100_10",
    #         chunk_size=100,
    #         chunk_overlap=10),
    # ).run()
    #
    # ExperimentPipeline(
    #     name="openai_small_1000_100_v1",
    #     llm=OpenAIAgent(),
    #     repo=DataRepository(
    #         embedding=OpenAiEmbeddingProvider(model="text-embedding-3-small"),
    #         db_path="./data/db/open_ai_large_1000_100",
    #         path="./data/r2.0/pdfs",
    #         name="open_ai_large_1000_100",
    #         chunk_size=1000,
    #         chunk_overlap=100),
    # ).run()
    #
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
