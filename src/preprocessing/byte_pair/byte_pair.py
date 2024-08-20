from typing import Dict, List, Tuple

from tqdm import tqdm


def fill_pairs(corpus_copy: List[Tuple[List[str], int]]) -> Dict[str, int]:
    pairs: Dict[str, int] = {}
    for word, nth in corpus_copy:
        j = 0
        while j < len(word) - 1:
            token = word[j] + word[j + 1]
            if token in pairs:
                pairs[token] += 1 * nth
            else:
                pairs[token] = 1 * nth
            j += 1
    return pairs


def search_max_iter(pairs: Dict[str, int]) -> str:
    max_iter = ""
    for x in pairs:
        if max_iter == "" or pairs[max_iter] < pairs[x]:
            max_iter = x
    return max_iter


def change_corpus(corpusCopy: List[Tuple[List[str], int]], token_to_merge: str) -> None:
    for i, (word, _) in enumerate(corpusCopy):
        j = 0
        while j < len(word) - 1:
            token = word[j] + word[j + 1]
            if token == token_to_merge:
                corpusCopy[i][0].pop(j)
                corpusCopy[i][0].pop(j)
                corpusCopy[i][0].insert(j, token)
            j += 1


def byte_pair(corpus: List[Tuple[List[str], int]], k: int) -> List[str]:
    corpus_copy: List[Tuple[List[str], int]] = corpus.copy()
    vocs: List[str] = list(set([c for word, _ in corpus_copy for c in word]))
    for _ in tqdm(range(k), desc="Byte_pair"):
        pairs: Dict[str, int] = fill_pairs(corpus_copy)
        max_iter: str = search_max_iter(pairs)
        change_corpus(corpus_copy, max_iter)
        vocs.insert(0, max_iter)
    return vocs
