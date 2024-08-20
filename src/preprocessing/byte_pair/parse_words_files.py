from collections import Counter
from typing import List, Tuple

from tqdm import tqdm


def serparate_text_to_word(text: str) -> List[str]:
    words: List[str] = []
    for word in text.split(" "):
        left = 0
        for right, _ in enumerate(word):
            if not word[right].isalnum():
                if right != left:
                    words.append(word[left:right])
                words.append(word[right])
                left = right + 1
            right += 1
        if left < len(word):
            words.append(word[left:])
    return words


def words_of_folder(files: List[str]) -> List[str]:
    words: List[str] = []
    for file in files:
        with open(file, "r", encoding="utf-8") as reader:
            words.extend(word for word in serparate_text_to_word(reader.read()))
    return words


def create_corpus(files: List[str]) -> List[Tuple[List[str], int]]:
    result: List[Tuple[List[str], int]] = []
    words: List[str] = words_of_folder(files)
    dictionary = Counter(words)
    for key in tqdm(dictionary, desc="create corpus"):
        result.append((list(key), dictionary[key]))
    return result
