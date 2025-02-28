'''
number - only a metric number is expected as an answer. No decimal commas or separators. Correct: 122333, incorrect: 122k, 122 233
name - only name is expected as an answer.
names - multiple names
boolean - only yes or no (or true, false). Case doesn't matter.
Important! Each schema also allows N/A or n/a which means "Not Applicable" or "There is not enough information even for a human to answer this question".

Adding the optional question_text element to Answer will allow the submission API to check if your answers are in the right order for correct scoring.
'''


import re

import json
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal, Self


class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(..., description="Physical page number in the PDF file")

#TODO wait for answer from organizator about data type for question number. What it sgould be integer or float
class Answer(BaseModel):
    question_text: Optional[str] = Field(None, description="Text of the question")
    kind: Optional[Literal["number", "name", "boolean", "names"]] = Field(None, description="Kind of the question")
    value: Union[float, str, bool, List[str], Literal["N/A"]] = Field(..., description="Answer to the question, according to the question schema")
    references: List[SourceReference] = Field([], description="References to the source material in the PDF file")


class AnswerSubmission(BaseModel):
    team_email: str = Field(..., description="Email that your team used to register for the challenge")
    submission_name: str = Field(..., description="Unique name of the submission (e.g. experiment name)")
    answers: List[Answer] = Field(..., description="List of answers to the questions")


class SubmissionParser:
    expected_types = {
        "name": str,
        "names": list,
        "boolean": (bool, str),
        "number": int
    }

    def __init__(self):
        self.all_data: List[Answer] = []

    @classmethod
    def from_raw_data(cls, raw_data: List[dict]) -> Self:
        """Создаёт объект SubmissionParser из списка сырых данных."""
        parser = cls()
        for item in raw_data:
            parser.append_new_question(
                question=item["question"],
                kind=item["kind"],
                answer=item["answer"],
                references=item["references"]
            )
        return parser

    def append_new_question(
        self, 
        question: str, 
        kind: str, 
        answer: Union[str, int, bool, list], 
        references: List[str]
    ) -> None:
        if answer in {'N/A', 'False'}:
            self.all_data.append(Answer(question_text=question, kind=kind, value=answer))
            return

        answer = self.process_answer(kind, answer)
        validated_references = self.process_references(references)
        
        self.all_data.append(Answer(
            question_text=question,
            kind=kind,
            value=answer,
            references=validated_references
        ))

    def process_answer(self, kind: str, answer: Union[str, int, bool, list]) -> Union[str, int, bool, list]:
        match kind:
            case "number":
                return self.process_number(answer)
            case "names":
                return self.process_names(answer)
            case "name":
                return self.process_name(answer)
            case "boolean":
                return self.process_boolean(answer)
            case _:
                raise ValueError(f"Unknown question type: {kind}")

    @staticmethod
    def process_number(answer: Union[str, int]) -> int:
        try:
            return int(float(answer))
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert answer {answer} to a number.")

    @staticmethod
    def process_names(answer: Union[str, list]) -> list:
        if isinstance(answer, str):
            return [name.strip() for name in answer.split(',')]
        if isinstance(answer, list) and all(isinstance(name, str) for name in answer):
            return answer
        raise TypeError("Expected a string or list for 'names' type.")

    @staticmethod
    def process_name(answer: str) -> str:
        if not isinstance(answer, str) or not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", answer.strip()):
            raise ValueError(f"Invalid name: {answer}. Only alphabetic characters, hyphens, and apostrophes are allowed.")
        return answer.strip()

    @staticmethod
    def process_boolean(answer: Union[str, bool]) -> bool:
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str) and answer.lower() in {"yes", "no", "true", "false"}:
            return answer.lower() in {"yes", "true"}
        raise ValueError(f"Invalid boolean value: {answer}. Expected 'yes'/'no' or 'true'/'false'.")

    @staticmethod
    def process_references(references: List[str]) -> List[SourceReference]:
        if len(references) < 2:
            raise ValueError("The list of references must contain at least two elements: sha and page_index.")

        sha, page_index = references[:2]
        if not isinstance(sha, str) or sha == 'N/A':
            raise TypeError(f"Invalid sha type: {sha}. Expected a string.")
        try:
            page_index = int(page_index)
        except (ValueError, TypeError):
            raise TypeError(f"Invalid page_index type: {page_index}. Expected an integer.")

        return [SourceReference(pdf_sha1=sha, page_index=page_index)]


if __name__ == "__main__":
    parser = SubmissionParser()
    answ = [
        ["According to the annual report, what is the Operating margin (%) for Altech Chemicals Ltd  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        'number',
        'N/A',
        'N/A',
        'N/A'],
        ["According to the annual report, what is the Operating margin (%) for Cofinimmo  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        'number',
        '81.0',
        '32',
        '9cc771c2171bacc138cda4e7d68b8b427a514d81/A'],
        ["What was the Capital expenditures (in USD) for Charles & Colvard, Ltd. according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        'number',
        '1496471.0',
        '70',
        'd3a834539b046a49708161a6c0d35aad29dd15ec'],
        ["According to the annual report, what is the Total revenue (in USD) for Winnebago Industries, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        'number',
        '4957730.0',
        '21',
        '7820b6e9487202b30f2883a6df91ae76f9461f2f'],
        ["According to the annual report, what is the Total revenue (in USD) for Lipocine Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        'number',
        '500000.0',
        '70',
        'c51f3c5aff7bea6fbc0bb7537838fa2f44f35c23'],
        ['Did HV Bancorp, Inc. mention any mergers or acquisitions in the annual report?',
        'boolean',
        'True',
        '4',
        '69a9dcb0bb6a46e2ff9f969d035e1774a2d49ef1'],
        ['Did Canadian Tire Corporation announce a share buyback plan in the annual report?',
        'boolean',
        'True',
        '21',
        '7c55d7900a241e732c145687598d43c915a678f9'],
        ["Which leadership **positions** changed at Canadian Tire Corporation in the reporting period? If data is not available, return 'N/A'.",
        'names',
        "'Martha', 'Owen'",
        '3',
        '7c55d7900a241e732c145687598d43c915a678f9'],
        ['Did Cofinimmo announce any changes to its dividend policy in the annual report?',
        'boolean',
        'False',
        'N/A',
        'N/A'],
        ["What was the value of Healthcare plan memberships (if applicable) of Nevro Corp. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        'number',
        'N/A',
        'N/A',
        'N/A'],
        ['What was the largest single spending of Maxeon Solar Technologies, Ltd. on executive compensation in USD?',
        'name',
        'N/A',
        'N/A',
        'N/A']]
    for i in answ:
        parser.append_new_question(i[0], i[1], i[2], [i[4],i[3]])

        
    with open("output.json", "w", encoding="utf-8") as file:
        json.dump([answer.model_dump() for answer in parser.all_data], file, ensure_ascii=False, indent=4)

