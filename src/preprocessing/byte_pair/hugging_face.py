import os
from typing import List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from . import MyBpeTokenizer


class MyBpeTokenizerHuggingFace(MyBpeTokenizer):

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def create_tokenizer(initial_corpus: List[str]) -> MyBpeTokenizer:

        if any(
            (not os.path.exists(path) or not os.path.isfile(path))
            for path in initial_corpus
        ):
            raise ValueError("'initial_corpus' must be a list of file path")

        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # And then train
        trainer = trainers.BpeTrainer(
            vocab_size=500,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )

        tokenizer.train(
            initial_corpus,
            trainer=trainer,
        )

        return MyBpeTokenizerHuggingFace(tokenizer)

    @staticmethod
    def load_from_file(path_file: str) -> MyBpeTokenizer:
        return MyBpeTokenizerHuggingFace(Tokenizer.from_file(path_file))

    def save(self, path_file: str) -> None:
        if not path_file.endswith("json"):
            raise ValueError("Must be a JSON file")
        self.tokenizer.save(path_file, pretty=True)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer
