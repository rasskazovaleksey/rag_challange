import json
import re
from typing import List, Tuple

from langchain_core.documents import Document

from ExperimentPipeline import ExperimentPipeline
from lib.Agent import IBMWatsonAgent
from lib.DataRepository import DataRepository
from lib.EmbeddingProvider import WatsonEmbeddingProvider
from lib.SubmissionSchema import SubmissionParser, SourceReference, AnswerSubmission


class ExperimentPipelineV2(ExperimentPipeline):

    def __init__(self, name, llm, repo):
        super().__init__(name, llm, repo)
        self.repoSmall = DataRepository(
            embedding=WatsonEmbeddingProvider(),
            db_path="./data/db/watson_ai_large_100_10_filtered",
            path="./data/r2.0/pdfs",
            name="watson_ai_large_100_10_filtered",
        )
        self.repoBig = DataRepository(
            embedding=WatsonEmbeddingProvider(),
            db_path="./data/db/watson_ai_large_1000_100_filtered",
            path="./data/r2.0/pdfs",
            name="watson_ai_large_1000_100_filtered",
        )

    @staticmethod
    def mapper(x):
        try:
            pattern = r'(?<!\d)(?:0?\.\d+|1(?:\.0+)?)(?!\d)'
            matches = re.findall(pattern, x['relevance'])
            matched = [float(match) for match in matches][0]
            return {"page": x['page'], "relevance": matched}
        except:
            print(f"Error in mapper {x}")
            return {"page": x['page'], "relevance": 0.0}

    @staticmethod
    def filter_function(x):
        try:
            return x['relevance'] >= 0.5
        except:
            return False

    def read_markdown(self, sha1, candidates, path_to_markdowns: str = './data/r2.0/markdown/'):
        with open(f'{path_to_markdowns}{sha1}/{sha1}.md', 'r', encoding='UTF-8') as file:
            data = file.read()
            parts = re.split(r'\{\d+\}-+', data)[1:]  # slice -1 page. because split goes to the beginning of the page

        pages_number = [p for p, _ in candidates]

        rag: List[Tuple[Document, float]] = []

        for number in pages_number:
            doc = (Document(page_content=parts[number]), 0.0)
            doc[0].metadata['page'] = doc[0].metadata['id'] = number
            doc[0].metadata['page_label'] = str(number + 1)
            doc[0].metadata['sha1'] = sha1

            rag.append(doc)

        return rag

    def check_answers(self):
        print(f"Running check_answers {self.name}")
        extracts = [self.extract(q) for q in self.questions]

        parser = SubmissionParser()

        with open("output_v2/answers.json", "r") as f:
            answers = json.load(f)
        with open("output_v2/relevance.json", "r") as f:
            relevances = json.load(f)

        for i, e in enumerate(extracts):
            try:
                answer = answers[e['original_question']]
            except KeyError:
                print("KeyError")
                continue

            if isinstance(answer, list):
                if len(answer) == 0:
                    raise ValueError(f"Answer is empty for {e['original_question']}")
                elif len(answer) == 1:
                    if answer[0]['answer'] == "False":
                        parser.append_new_question(
                            question=e['original_question'],
                            kind=e['type'],
                            answer=answer[0]['answer'],
                            references=[],
                        )
                    else:
                        parser.append_new_question(
                            question=e['original_question'],
                            kind=e['type'],
                            answer=answer[0]['answer'],
                            references=[e["sha1"], answer[0]['page']],
                        )

                else:
                    if e['comparison'] == None:
                        unique = set([a['answer'] for a in answer])
                        if len(unique) == 1:
                            refs = [SourceReference(pdf_sha1=e["sha1"], page_index=a['page']) for a in answer]
                            parser.append_new_question_ref(
                                question=e['original_question'],
                                kind=e['type'],
                                answer=answer[0]['answer'],
                                references=refs,
                            )
                        else:
                            filtered = list(filter(lambda x: x['answer'] != "N/A", answer))
                            filtered = list(filter(lambda x: x['answer'] != "False", filtered))
                            unique = set([a['answer'] for a in filtered])
                            if len(unique) == 1:
                                refs = [SourceReference(pdf_sha1=e["sha1"], page_index=a['page']) for a in filtered]
                                parser.append_new_question_ref(
                                    question=e['original_question'],
                                    kind=e['type'],
                                    answer=list(unique)[0],
                                    references=refs,
                                )
                            else:
                                # TODO: Collect rechecked answers, sould be one more call to LLM
                                # rel = relevances[e['original_question']]
                                # docs = self.read_markdown(e['sha1'], [(r['page'], r['relevance']) for r in rel])
                                # answer = self.llm.query(
                                #     text=e['original_question'],
                                #     data=docs,
                                #     path=f"./prompt/{e['type']}_prompt.txt",
                                #     system="You are competent financial analytic.")
                                # print(
                                #     {
                                #         "question": e['original_question'],
                                #         "answer": answer,
                                #     }
                                # )
                                with open("output_v2/rechecked_answer.json", "r") as f:
                                    rechecked = json.load(f)

                                rech = list(filter(lambda x: x['question'] == e['original_question'], rechecked))[0]
                                filtered_rech = list(filter(lambda x: x['answer'] == rech['answer'], answer))
                                if len(filtered_rech) == 0:
                                    rm_na = list(filter(lambda x: x['answer'] != "N/A", answer))
                                    refs = [SourceReference(pdf_sha1=e["sha1"], page_index=a['page']) for a in rm_na]
                                    parser.append_new_question_ref(
                                        question=e['original_question'],
                                        kind=e['type'],
                                        answer=rech['answer'],
                                        references=refs,
                                    )
                                else:
                                    refs = [SourceReference(pdf_sha1=e["sha1"], page_index=a['page']) for a in filtered_rech]
                                    parser.append_new_question_ref(
                                        question=e['original_question'],
                                        kind=e['type'],
                                        answer=rech['answer'],
                                        references=refs,
                                    )
                    else:
                        # TODO: Collect rechecked answers, sould be one more call to LLM
                        docs = []
                        for a in answer:
                            filtered = list(filter(lambda x: x['answer'] != "N/A", a['answers']))
                            filtered = list(filter(lambda x: x['answer'] != "", filtered))
                            for f in filtered:
                                doc = (Document(page_content=f"{a['company']} has {e['metric']} {f['answer']}"), 0.0)
                                doc[0].metadata['page'] = f['page']
                                doc[0].metadata['sha1'] = f['sha1']
                                docs.append(doc)
                        # answer = self.llm.query(
                        #     text=e['original_question'],
                        #     data=docs,
                        #     path=f"./prompt/{e['type']}_prompt.txt",
                        #     system="You are competent financial analytic.")
                        # print(
                        #     {
                        #         "question": e['original_question'],
                        #         "answer": answer,
                        #     }
                        # )
                        with open("output_v2/comparison_check.json", "r") as f:
                            rechecked = json.load(f)

                        rech = list(filter(lambda x: x['question'] == e['original_question'], rechecked))[0]
                        refs = [SourceReference(pdf_sha1=a[0].metadata['sha1'], page_index=a[0].metadata['page']) for a in docs]
                        parser.append_new_question_ref(
                            question=e['original_question'],
                            kind=e['type'],
                            answer=rech['answer'],
                            references=refs,
                        )

            else:
                if answer['answer'] is None:
                    parser.append_new_question(
                        question=e['original_question'],
                        kind=e['type'],
                        answer="N/A",
                        references=[],
                    )
                else:
                    raise ValueError(f"Answer is not None for {e['original_question']}")

        print(len(parser.all_data))
        submition = AnswerSubmission(
            answers=parser.all_data,
            team_email="xxx.xxx@gmail.com",
            submission_name="pjatk_team_002"
        )
        with open("pjatk_team_002.json", "w", encoding="utf-8") as file:
            json.dump(submition.model_dump(), file, ensure_ascii=False, indent=4)

    def create_answers(self):
        print(f"Running create_answers {self.name}")
        print(f"Running save_relevance {self.name}")
        extracts = [self.extract(q) for q in self.questions]
        with open("./output_v2/relevance.json", "r") as f:
            relevance = json.load(f)
        for i, e in enumerate(extracts):
            print(f"Processing {i}/{len(extracts) - 1} with sha1 {e['sha1']}")
            # print(e['original_question'])
            relevant = relevance[e['original_question']]

            with open("./output_v2/answers.json", "r") as f:
                j = json.load(f)
                if e['original_question'] in j:
                    print(f"Skipping {e['original_question']}")
                    continue

            if len(e['companies']) == 1:
                print(relevant)
                if len(relevant) == 0:
                    print("No relevance found")
                    j[e['original_question']] = {
                        'answer': None,
                        'page': None
                    }
                    json.dump(j, open("./output_v2/answers.json", "w"), ensure_ascii=False, indent=4)
                else:
                    answers = []
                    for r in relevant:
                        print(e, r)
                        d = self.read_markdown(e['sha1'], [(r['page'], r['relevance'])])
                        answer = self.llm.query(
                            text=e['original_question'],
                            data=d,
                            path=f"./prompt/{e['type']}_prompt.txt",
                            system="You are competent financial analytic.")
                        answers.append({
                            "sha1": e['sha1'],
                            "page": r['page'],
                            "answer": answer
                        })
                    j[e['original_question']] = answers
                    print(answers)
                    json.dump(j, open("./output_v2/answers.json", "w"), ensure_ascii=False, indent=4)

            else:
                print(e)
                print("!!!!")
                relevant = relevance[e['original_question']]
                j[e['original_question']] = []
                for company, values in relevant.items():
                    print(company, values)
                    l = list(filter(lambda x: company in x['company_name'], self.subset))[0]
                    sha1 = l['sha1']
                    print(sha1)
                    answers = []
                    for v in values:
                        print(v)
                        d = self.read_markdown(sha1, [(v['page'], v['relevance'])])
                        q = f"What is the {e['metric']} in {e['currency']}?"
                        print(q)
                        answer = self.llm.query(
                            text=q,
                            data=d,
                            path=f"./prompt/number_prompt.txt",
                            system="You are competent financial analytic.")
                        print(answer)
                        answers.append({
                            "sha1": sha1,
                            "page": v['page'],
                            "answer": answer
                        })
                    j[e['original_question']].append({
                        'company': company,
                        'answers': answers
                    })
                    print(j[e['original_question']])
                json.dump(j, open("./output_v2/answers.json", "w"), ensure_ascii=False, indent=4)

    def save_relevance(self):
        print(f"Running save_relevance {self.name}")
        extracts = [self.extract(q) for q in self.questions]
        synonyms_lookup = self.read_synonyms()
        for i, e in enumerate(extracts):
            print(f"Processing {i}/{len(extracts) - 1} with sha1 {e['sha1']}")
            if len(e['companies']) == 1 or len(e['companies']) == 4 or len(e['companies']) == 5 or len(
                    e['companies']) == 6:
                pass
            else:
                raise ValueError(f"Companies is {len(e['companies'])} for {e}")

            # if e['companies'] == ['4d3e52b69b4b5366e54ce87cf641b01b1419bdee',
            #                       '553afbf09b6d83166b17acb02431c6cf38e4defc',
            #                       '980742aa08ea64d552c153bcefbd7e8243fb9efd',
            #                       'cc0fc5888b99758100a7ff024863fc4337b6b3c5']:
            #     continue
            # if e['sha1'] != "aa781901e117281bfee6f8e4bea6fc9c9bada62e":
            #     continue

            with open("./output_v2/relevance.json", "r") as f:
                j = json.load(f)
                if e['original_question'] in j:
                    print(f"Skipping {e['original_question']}")
                    continue

            print(e)
            if len(e['companies']) == 1:
                synonyms = list(filter(lambda x: x['metric'] == e['metric'], synonyms_lookup))[0]

                self.repo = self.repoSmall
                search = self.search_database(synonyms, e, main=10, side=5)
                smallCandidates = self.filter_candidates(search, size=10)

                self.repo = self.repoBig
                search = self.search_database(synonyms, e, main=10, side=5)
                bigCandidates = self.filter_candidates(search, size=10)

                mergedCandidates = self.merge_data(smallCandidates + bigCandidates)[0:10]
                documents = self.read_pdf(e['sha1'], mergedCandidates)
                assert len(documents) == len(mergedCandidates)

                relevance_holder = []
                for d in documents:
                    text = f"Evaluate the context for its relevance to the question: '{e['original_question']}'."
                    relevance_answer = self.llm.query(
                        text=text,
                        data=[d],
                        path=f"./prompt/relevance_prompt.txt",
                        system="You are competent financial analytic.")
                    answer = {"page": d[0].metadata['page'], "relevance": relevance_answer}
                    print(answer)
                    relevance_holder.append(answer)

                filtered_scores = list(filter(self.filter_function, list(map(self.mapper, relevance_holder))))
                print(filtered_scores)
                with open("./output_v2/relevance.json", "r") as f:
                    j = json.load(f)
                j[e['original_question']] = filtered_scores
                json.dump(j, open("./output_v2/relevance.json", "w"), ensure_ascii=False, indent=4)
            else:
                print(f"Comparison problem found for {e['companies']}")
                holder = {}
                for c in e['companies']:
                    if c == 'Inc.':
                        print("--- Inc. found, skipping its and error in the code. ---")
                        continue
                    l = list(filter(lambda x: c in x['company_name'], self.subset))
                    sha1 = l[0]['sha1']
                    copy_e = e.copy()
                    copy_e['sha1'] = sha1
                    copy_e['original_question'] = f"{e['metric']} of {c}"

                    synonyms = list(filter(lambda x: x['metric'] == e['metric'], synonyms_lookup))[0]

                    self.repo = self.repoSmall
                    search = self.search_database(synonyms, copy_e, main=10, side=5)
                    smallCandidates = self.filter_candidates(search, size=10)

                    self.repo = self.repoBig
                    search = self.search_database(synonyms, copy_e, main=10, side=5)
                    bigCandidates = self.filter_candidates(search, size=10)

                    mergedCandidates = self.merge_data(smallCandidates + bigCandidates)[0:10]

                    documents = self.read_pdf(copy_e['sha1'], mergedCandidates)
                    assert len(documents) == len(mergedCandidates)

                    relevance_holder = []
                    for d in documents:
                        text = f"Evaluate the context for its relevance to the question: '{copy_e['original_question']}'."
                        relevance_answer = self.llm.query(
                            text=text,
                            data=[d],
                            path=f"./prompt/relevance_prompt.txt",
                            system="You are competent financial analytic.")
                        answer = {"page": d[0].metadata['page'], "relevance": relevance_answer}
                        print(answer)
                        relevance_holder.append(answer)

                    filtered_scores = list(filter(self.filter_function, list(map(self.mapper, relevance_holder))))
                    holder[c] = filtered_scores
                    print(holder[c])
                    print(filtered_scores)

                print(holder)
                with open("./output_v2/relevance.json", "r") as f:
                    j = json.load(f)
                j[e['original_question']] = holder
                json.dump(j, open("./output_v2/relevance.json", "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pipeline = ExperimentPipelineV2(
        name="watson_embedding_deepseek_r1_distill_llama_70b",
        llm=IBMWatsonAgent(model="deepseek/deepseek-r1-distill-llama-70b"),
        repo=None,
    )
    # pipeline.save_relevance()
    # pipeline.create_answers()
    pipeline.check_answers()
