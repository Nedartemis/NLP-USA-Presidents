import nltk
import pandas as pd
import tiktoken
from nltk.lm import MLE, StupidBackoff, Laplace
from nltk import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

class MyNgramModel():
    def __init__(self,
        n: int = 2,
        language_model: callable = Laplace,
        tokenizer: callable = word_tokenize,
        detokenizer: callable = None
        ) -> None:
        """
        MyNgramModel init function
        :param n: the order of the n gram
        :param language_model: the chosen language model (MLE, StupidBackoff, Laplace)
        :param tokenizer: the tokenizer method
        :param detokenizer: the detokenizer method
        """
        # Initialize language model
        self.n = n
        self.lm = language_model(n) if language_model != StupidBackoff else language_model(order = n)

        # With an encoding tokenizer -> specify tokenizer and detokenizer (to decode the generated text)
        # With an non encoding tokenizer -> specify only the tokenizer (the generated text will still be strings)
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

        # Using the model's methods
        self.fit = self.lm.fit
        self.perplexity = self.lm.perplexity
        self.score = self.lm.score
        self.counts = self.lm.counts
        self.vocab = self.lm.vocab
        self.lookup = self.lm.vocab.lookup

    def preprocessing(self,
        df: pd.DataFrame,
        lowercase: bool = True,
        strip_quotes : bool = True) -> list[str] | str:
        """
        preprocess a dataframe before the learning phase
        :param df: the dataset's dataframe
        :param lowercase: boolean transform the string to lowercase
        :param strip_quotes: boolean to strip quotes (at the beginning and end of the string)
        :return: the preprocessed dataframe as a list of string or string
        """

        train_string = ' '.join(df.values.flatten())
        if lowercase:
            train_string = train_string.lower()
        if strip_quotes:
            train_string = train_string.strip('"')

        train_sents = sent_tokenize(train_string)
        return [self.tokenizer(s) for s in train_sents]
    
    def format_result_string(self, result_string: str) -> str:
        """
        format the generated string
        :param result_string: the string to format
        :return: the generated text as a readable string
        """
        formated_URL =  result_string.replace("tokurl", "**URL**")
        return formated_URL
    
    def generate(self,
        num_words: int,
        text_seed: list[int] = []) -> str:
        """
        generates num_words words
        :param num_words: the number of words to generate
        :param text_seed: the context given to the model to generate
        :return: the generated text as a readable string
        """
        generated_text = self.lm.generate(num_words, text_seed=text_seed)
        result_string = ""
        if self.detokenizer != None:
            result_string = self.detokenizer(generated_text)
        else:
            dropped_tags = [s for s in generated_text if s not in ["<s>", "</s>"]]
            result_string = " ".join(dropped_tags)
        return self.format_result_string(result_string)
            
