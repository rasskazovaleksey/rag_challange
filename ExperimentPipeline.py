import json
import re
from os import walk

from langchain_community.document_loaders import PyPDFLoader

from lib.Agent import IBMWatsonAgent, Agent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import WatsonEmbeddingProvider
from lib.questions import QuestionExtractor


class ExperimentPipeline:
    def __init__(self,
                 questions_path="data/r2.0/questions.json",
                 subset_path="data/r2.0/subset.json",
                 llm: Agent = IBMWatsonAgent(model="meta-llama/llama-3-405b-instruct"),
                 repo=DataRepository(
                     embedding=WatsonEmbeddingProvider(),
                     db_path="./data/db/watson_ai_large_100_10_filtered",
                     path="./data/r2.0/pdfs",
                     name="watson_ai_large_100_10_filtered",
                     chunk_size=100,
                     chunk_overlap=10),
                 ):
        self.extractor = QuestionExtractor()
        self.questions = self.read_questions(questions_path)
        self.subset = self.read_subset(subset_path)
        self.llm = llm
        self.repo = repo

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
        extract['sha1'] = list(filter(lambda x: x["company_name"] in extract['companies'], self.subset))[0]['sha1']
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
        main_results = self.repo.query(synonyms['metric'], k=main,
                                       f=sha_filter)  # start with main metric from the question
        smaller_results = []  # start with main metric from the question
        for m in synonyms['synonyms']:
            smaller_results += self.repo.query(m['text'], k=side)  # find similar metrics
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

    def read_pdf(self, sha1, candidates):
        document_loader = PyPDFLoader(f"./data/r2.0/pdfs/{sha1}.pdf")
        doc = document_loader.load()
        pages_number = [p for p, _ in candidates]
        print(candidates)
        print(pages_number)
        print(doc[100])
        # pages = [p for p in doc if p.metadata["page"] in pages_number]
        # for p in pages:
        #     print(p.metadata["page"])
        #     p.metadata["id"] = p.metadata["page"]

    def run(self):
        extracts = [self.extract(q) for q in self.questions]
        synonyms_lookup = self.read_synonyms()
        for i, e in enumerate(extracts):
            print(f"Processing {i}/{len(extracts) - 1} with sha1 {e['sha1']}")
            if len(e['companies']) == 1 or len(e['companies']) == 4 or len(e['companies']) == 5 or len(
                    e['companies']) == 6:
                pass
            else:
                raise ValueError(f"Companies is {len(e['companies'])} for {e}")
            if len(e['companies']) > 1:
                print(f"Comparison problem found")

            if len(e['companies']) == 1:
                synonyms = list(filter(lambda x: x['metric'] == e['metric'], synonyms_lookup))[0]
                all_candidates = self.search_database(synonyms, e, main=10, side=5)
                candidates = self.filter_candidates(all_candidates, size=8)
                documents = self.read_pdf(e['sha1'], candidates)
            else:
                print(f"Comparison problem found")


if __name__ == "__main__":
    ep = ExperimentPipeline()
    ep.run()
    # print(ep.get_synonyms("Operational margin"))
