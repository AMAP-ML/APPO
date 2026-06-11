import sys
import os

sys.path.append(os.getcwd())

import json
from typing import Any, Dict, List, Optional, Tuple

from .utils import extract_solution, last_boxed_only_string, remove_boxed


class DataLoader:
    """Loading Datasets"""

    def __init__(self, dataset_name, data_path: str):
        """
        Args:
            dataset_name
            data_path
        """
        self.dataset_name = dataset_name
        self.data_path = os.path.join(data_path, dataset_name, 'test.jsonl')

    def load_data(self) -> Tuple[List[str], List[str], List[Optional[Dict[str, Any]]]]:
        """
        Load dataset

        Returns:
            (questions, answers, metadata_list)
            metadata_list contains extra fields (e.g. Level) for each sample, or None if not available.
        """
        questions = []
        answers = []
        metadata_list = []

        print(f"Loading dataset from {self.data_path}")

        if (
            "aime24" in self.data_path
            or "amc23" in self.data_path
            or "gsm8k" in self.data_path
            or "tabmwp" in self.data_path
            or "gaokao2023en" in self.data_path
            or "college_math" in self.data_path
        ):
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answer = data["answer"]
                    if "gsm8k" in self.data_path:
                        answer = extract_solution(answer)
                    answers.append(answer)
        elif "svamp" in self.data_path or "asdiv" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    body = data["body"] if "body" in data else data["Body"]
                    question = (
                        data["question"] if "question" in data else data["Question"]
                    )
                    answer = data["answer"] if "answer" in data else data["Answer"]
                    if "asdiv" in self.data_path:
                        answer = answer.split(" (")[0]
                    questions.append(body + " " + question)
                    answers.append(answer)
        elif "mawps" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["input"])
                    answers.append(data["target"])
        elif "carp_en" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["content"])
                    answers.append(data["answer"])
        elif "minerva_math" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    question = data["problem"]
                    answer = data["solution"]
                    try:
                        answer = remove_boxed(last_boxed_only_string(answer))
                    except:
                        pass
                    questions.append(question)
                    answers.append(answer)
        elif "olympiadbench" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["final_answer"][0])
        elif "/math/test" in self.data_path or "aime25" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["problem"])
                    answers.append(data["answer"])
        elif "gaia" in self.data_path:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["Question"])
                    answers.append(data["answer"])
                    metadata_list.append({"Level": data.get("Level")})
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    questions.append(data["question"])
                    answers.append(data["answer"])

        # Pad metadata_list to match questions length if not populated
        while len(metadata_list) < len(questions):
            metadata_list.append(None)

        print(f"Loading {len(questions)} samples from {self.data_path}...")
        return questions, answers, metadata_list
