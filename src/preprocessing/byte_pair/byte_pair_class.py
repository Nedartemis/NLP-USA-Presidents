from typing import List, Tuple

from tqdm import tqdm

from src.preprocessing.byte_pair import MyBpeTokenizer
from src.preprocessing.byte_pair.byte_pair import byte_pair
from src.preprocessing.byte_pair.parse_words_files import (
    create_corpus,
    serparate_text_to_word,
)


class MyBpeTokenizerBytePair(MyBpeTokenizer):
    def __init__(self, path_tokenizer: str) -> None:
        self.path_tokenizer = path_tokenizer

    @staticmethod
    def create_tokenizer(initial_corpus: List[str]) -> MyBpeTokenizer:
        corpus: List[Tuple[List[str], int]] = create_corpus(initial_corpus)
        vocs: List[str] = byte_pair(corpus, 600)
        path: str = "./tokenizer.txt"
        with open(path, "w", encoding="utf-8") as writer:
            for word in tqdm(vocs, desc="write into file"):
                writer.write(word + "\n")
        return MyBpeTokenizerBytePair(path)

    @staticmethod
    def load_from_file(path_file: str) -> MyBpeTokenizer:
        return MyBpeTokenizerBytePair(path_file)

    def save(self, path_file: str) -> None:
        self.path_tokenizer = path_file

    def tokenize(self, text: str) -> List[str]:
        result: List[str] = []
        with open(self.path_tokenizer, encoding="utf-8") as f:
            tokenizer: List[str] = f.read().splitlines()
            words: List[str] = serparate_text_to_word(text)
            for word in words:
                n = len(word)
                begin = 0
                end = 1
                while end <= n:
                    while end <= n and word[begin : end + 1] in tokenizer:
                        end += 1
                    result.append(word[begin:end])
                    begin = end
                    end += 1
                result.append(" ")
        if result[-1] == " ":
            result.pop()

        return result

    def get_tokenizer(self) -> str:
        return self.path_tokenizer
