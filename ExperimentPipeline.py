import json
import re
from os import walk

from lib.Agent import IBMWatsonAgent
from lib.questions import QuestionExtractor

class ExperimentPipeline:
    def __init__(self, questions_path="data/r2.0/questions.json", subset_path="data/r2.0/subset.json"):
        self.extractor = QuestionExtractor()
        self.questions = self.read_questions(questions_path)
        self.subset = self.read_subset(subset_path)
        self.llm = IBMWatsonAgent(model="meta-llama/llama-3-405b-instruct")

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
                text = text.replace("\"Here are 5 synonimatic/similar expressions for the term \\\"total assets\\\":", '')
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

    def run(self):
        extracts = [self.extract(q) for q in self.questions]
        for e in extracts:
            if len(e['companies']) == 1 or len(e['companies']) == 4 or len(e['companies']) == 5 or len(
                    e['companies']) == 6:
                pass
            else:
                raise ValueError(f"Companies is {len(e['companies'])} for {e}")
            if len(e['companies']) > 1:
                print(f"Comparison problem found")


if __name__ == "__main__":
    ep = ExperimentPipeline()
    ep.read_synonyms()
    # print(ep.get_synonyms("Operational margin"))
