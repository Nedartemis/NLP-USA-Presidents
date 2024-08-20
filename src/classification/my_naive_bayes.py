from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from spacy.lang.en.stop_words import STOP_WORDS as en_stop


class MyNaiveBayes:

    def __init__(
        self, cv: Optional[CountVectorizer] = None, clf: Optional[MultinomialNB] = None
    ):
        self.cv = cv or CountVectorizer(ngram_range=(1, 1), stop_words=list(en_stop))
        self.clf = clf or MultinomialNB()
        self.model = make_pipeline(self.cv, self.clf)

        self.fit = self.model.fit
        self.predict = self.model.predict

    def get_most_important_token(self, n: int = 15) -> Dict[str, List[str]]:
        feature_names = np.asarray(self.cv.get_feature_names_out())
        res = {
            category: feature_names[
                np.argsort(self.clf.feature_log_prob_[i])[-n:][::-1]
            ].tolist()
            for i, category in enumerate(self.clf.classes_)
        }
        return res

    def predict_and_get_most_important_tokens(self, text: str) -> Tuple[str, List[str]]:
        raise NotImplementedError()

    def most_important_words(
        self, text: str, treshold: float = 0.9
    ) -> Dict[str, List[str]]:

        # representation of a text by a bag of words
        rep: np.ndarray = self.cv.transform([text]).toarray()

        # the coef of each words for each class
        coefs = self.clf.feature_count_

        prediction_value_for_each_class = (rep @ coefs.T)[0]

        rep = rep.reshape((rep.shape[1]))
        tokens = self.cv.get_feature_names_out()[rep != 0]
        important_words: Dict[str, List[str]] = {}

        for index_class, words_weight in enumerate(coefs * rep):
            assert isinstance(words_weight, np.ndarray)
            words_weight = words_weight[rep != 0]
            indexes = words_weight.argsort()[::-1]

            classp = self.clf.classes_[index_class]

            for i in range(len(indexes)):
                sub_indexes = indexes[: i + 1]

                # if left is higher it means that the first i words represent 0.9 (treshold) of the value predicted
                if (
                    words_weight[sub_indexes].sum()
                    > prediction_value_for_each_class[index_class] * treshold
                ):
                    important_words[classp] = tokens[sub_indexes]
                    break

        return important_words
