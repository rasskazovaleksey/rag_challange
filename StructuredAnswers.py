import json

from lib.Agent import OpenAIAgent, IBMWatsonAgent
from lib.questions import QuestionExtractor

questionsExtractor = QuestionExtractor()


def read_questions():
    with open('./data/r2.0/questions.json') as f:
        questions = json.load(f)
    return questions


def read_json_files():
    with open('openai_large_100_10_v1.json') as f:
        open_ai_large_100_10 = json.load(f)

    with open('openai_small_1000_100_v1.json') as f:
        open_ai_small_1000_100 = json.load(f)

    with open('openai_small_1000_100_filtered_v1.json') as f:
        open_ai_small_1000_100_filtered = json.load(f)

    with open('watson_small_llama_405b_v1.json') as f:
        watson_large_llama_405b = json.load(f)

    with open('watson_small_llama_405b_v1.json') as f:
        watson_small_llama_405b = json.load(f)

    return open_ai_large_100_10, open_ai_small_1000_100, open_ai_small_1000_100_filtered, watson_large_llama_405b, watson_small_llama_405b


def find_agents_opinion(a, b, c, d, e, question):
    first = list(filter(lambda x: x['extract']['original_question'] == question['text'], a))[0]
    second = list(filter(lambda x: x['extract']['original_question'] == question['text'], b))[0]
    third = list(filter(lambda x: x['extract']['original_question'] == question['text'], c))[0]
    fourth = list(filter(lambda x: x['extract']['original_question'] == question['text'], d))[0]
    fifth = list(filter(lambda x: x['extract']['original_question'] == question['text'], e))[0]

    assert first is not None, f"Could not find question {question['text']} in a"
    assert second is not None, f"Could not find question {question['text']} in b"
    assert third is not None, f"Could not find question {question['text']} in c"
    assert fourth is not None, f"Could not find question {question['text']} in d"
    assert fifth is not None, f"Could not find question {question['text']} in e"

    assert first['extract']['sha1'] == second['extract']['sha1'] == third['extract']['sha1'] == fourth['extract'][
        'sha1'] == fifth['extract']['sha1'], f"SHA1 mismatch for question {question['text']}"

    extract = questionsExtractor.extract(question['text'])
    if extract["comparison"] is None:
        return {
            'first': first['answer'],
            'second': second['answer'],
            'third': third['answer'],
            'fourth': fourth['answer'],
            'fifth': fifth['answer'],
        }
    else:
        return {
            'first': first['holder'],
            'second': second['holder'],
            'third': third['holder'],
            'fourth': fourth['holder'],
            'fifth': fifth['holder'],
        }


if __name__ == "__main__":
    a, b, c, d, e = read_json_files()
    questions = read_questions()

    holder = []

    for i, question in enumerate(questions):
        answers = find_agents_opinion(a, b, c, d, e, question)
        holder.append({
            'question': question['text'],
            'answers': answers
        })

    for h in holder:
        print(h['question'])
        if h['answers']['fourth'] == h['answers']['fifth']:
            print("Identical")
            print(h['answers']['fourth'], h['answers']['fifth'])
        else:
            print("Different")
            print(h['answers']['fourth'], h['answers']['fifth'])

    context = "\n\n---\n\n".join([f"{h}" for h in holder])
    prompt = """
    I have questions with 5 options to answer each question and reference page in brackets.
    Give me the best approximation of each answer and corresponding page for each question.
    Give answer in JSON format ONLY. Just Json in form, if you are not sure rely on fourth and fifth agent opinion.
    Return a JSON array where each object follows this format:
       [
           {{
               "question": "<QUESTION>",
               "answer": <answer>,
               "page": <PAGE>
           }}
       ]
    Provide no additional text or disclaimers in your response. Dont miss any question, output all 100.
    """
    # agent = OpenAIAgent(model="gpt-4.5-preview")
    agent = IBMWatsonAgent(model="meta-llama/llama-3-405b-instruct")
    for h in holder:
        q = str(h) + "\n" + prompt
        answer = agent.query(q, [], system="You researcher with greate capability.", path="./prompt/empty_prompt.txt")
        print(answer)
