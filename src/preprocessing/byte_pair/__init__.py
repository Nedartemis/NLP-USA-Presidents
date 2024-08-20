from abc import ABC, abstractmethod
from typing import List


class MyBpeTokenizer(ABC):

    @staticmethod
    @abstractmethod
    def create_tokenizer(initial_corpus: List[str]) -> "MyBpeTokenizer":
        """
        create tokenizer from Folder of files
        """

    @staticmethod
    @abstractmethod
    def load_from_file(path_file: str) -> "MyBpeTokenizer":
        """
        load tokenizer from a tokenizer_file
        """

    @abstractmethod
    def save(self, path_file: str) -> None:
        """
        save tokenizer in path file
        """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        tokenize sentence form tokenizer
        """
