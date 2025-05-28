import json

if __name__ == '__main__':
    print("Processing all questions")

    incorrect = 0
    correct = 0

    with open("correct_answers.json", "r") as f:
        correct_answers = json.load(f)

    for question in correct_answers:
        question_answers = correct_answers[question]["answers"]
        question = question.replace("'", "").replace('"', '').strip()

        with open("answer.json", "r") as f:
            model_answers = json.load(f)

        model_answers_filtered = list(filter(lambda x: x["question"].replace("'", "").replace('"', '') == question, model_answers))
        assert len(model_answers_filtered) == 1, f"Expected exactly one answer for question '{question}', found {len(model_answers_filtered)}"

        if model_answers_filtered[0]["answer"] in question_answers:
            correct += 1
        else:
            incorrect += 1

    print(f"Correct answers: {correct}")
    print(f"Incorrect answers: {incorrect}")
    print(f"Total questions processed: {correct + incorrect}")
    print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")
