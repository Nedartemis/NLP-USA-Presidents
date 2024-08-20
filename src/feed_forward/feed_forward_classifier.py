import os
import time
from typing import Callable, List

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer


def files_to_df(files: List[str]) -> pd.DataFrame:
    """
    return DataFrame with 2 columns:
        name: strings
        text: List(String)
    """
    list_of_dfs = []
    for file in files:
        df_tmp = pd.read_csv(file)[["name", "text"]]
        list_of_dfs.append(df_tmp)

    return pd.concat(list_of_dfs, ignore_index=True)


def vectorize_text(
    text,
    word2vec_model: Word2Vec,
    tokenizer: Callable[[str], List[str]] = word_tokenize,
):
    tokens = tokenizer(text)

    word_vectors = []
    for token in tokens:
        if token in word2vec_model.wv.key_to_index:
            word_vectors.append(word2vec_model.wv[token])

    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)

    text_vector = np.mean(word_vectors, axis=0)
    return text_vector


def clasification_from_model(
    text,
    model,
    word2vec_model: Word2Vec,
    tokenizer: Callable[[str], List[str]] = word_tokenize,
):
    tokens = tokenizer(text)
    word_vectors = []
    for token in tokens:
        if token in word2vec_model.wv.key_to_index:
            word_vectors.append(word2vec_model.wv[token])
    if not word_vectors:
        word_vectors = np.zeros(word2vec_model.vector_size)

    text_vector = np.mean(word_vectors, axis=0)
    text_vector = text_vector.reshape(1, -1)

    predictions = model.predict(text_vector)
    predict_class = np.argmax(predictions)
    return predict_class
