from datasets import load_dataset

class AIME24:
    """
    Loads AIME 2024 Benchmark
    Source: math-ai/aime24 (split: 'test')
    Schema usually: {'problem': str, 'solution': str, 'answer': str}
    """
    name = "AIME-24"

    def __init__(self, split: str = "test"):
        self.ds = load_dataset("math-ai/aime24", split=split)

    def __iter__(self):
        for ex in self.ds:
            # unify schema to {question, answer}
            yield {"question": ex.get("problem", ex.get("question")), 
                   "answer": ex["answer"]}


class Gaokao2023En:
    """
    Loads Gaokao 2023 Math (English version)
    Source: MARIO-Math-Reasoning/Gaokao2023-Math-En
    Schema: {'question': str, 'answer': str}
    """
    name = "Gaokao-2023-EN"

    def __init__(self, split: str = "test"):
        self.ds = load_dataset("MARIO-Math-Reasoning/Gaokao2023-Math-En", split=split)

    def __iter__(self):
        for ex in self.ds:
            yield {"question": ex["question"], "answer": ex["answer"]}


class MinervaMath:
    """
    Loads MinervaMath benchmark
    Source: math-ai/minervamath
    Schema: {'problem': str, 'answer': str, ...}
    """
    name = "MinervaMath"

    def __init__(self, split: str = "test"):
        self.ds = load_dataset("math-ai/minervamath", split=split)

    def __iter__(self):
        for ex in self.ds:
            yield {"question": ex.get("problem", ex.get("question",'')), 
                   "answer": ex["answer"]}
