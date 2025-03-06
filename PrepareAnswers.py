import json

from opentelemetry.propagate import extract

from lib.SubmissionSchema import SubmissionParser, AnswerSubmission


def read_questions():
    with open('./data/r2.0/questions.json') as f:
        questions = json.load(f)
    return questions

if __name__ == "__main__":
    with open("processed_answer.json", "r") as f:
        data = json.load(f)
    with open('./data/r2.0/questions.json') as f:
        questions = json.load(f)
    with open('watson_small_llama_405b_v1.json') as f:
        extracts = json.load(f)
    with open('./data/r2.0/subset.json') as f:
        subsets = json.load(f)

    parser = SubmissionParser()

    for d in data:
        question = list(filter(lambda x: x['text'] == d['question'], questions))
        extract = list(filter(lambda x: x['extract']['original_question'] == d['question'], extracts))
        if len(question) != 1:
            raise ValueError(f"Question not found for '{d}'")
        if len(extract) != 1:
            raise ValueError(f"Extract not found for '{d}'")
        assert question[0]['text'] == d['question']

        if extract[0]['extract']['comparison'] is not None:
            comp = list(filter(lambda x: x['company_name'] == d['answer'], subsets))[0]
            extract[0]['extract']['sha1'] = comp['sha1']

        ref = [extract[0]['extract']['sha1'], d['page']]

        parser.append_new_question(
            question=question[0]['text'],
            kind=question[0]['kind'],
            answer=d['answer'],
            references=ref
        )

    answers = parser.all_data
    print(len(answers))
    submition = AnswerSubmission(
        answers=answers,
        team_email="xxx.xxx@gmail.com",
        submission_name="pjatk_team_001"
    )
    print(len(submition.answers))

    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(submition.model_dump(), file, ensure_ascii=False, indent=4)