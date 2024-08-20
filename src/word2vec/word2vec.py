import time
from typing import Callable, List, Optional

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def files_to_df(files: List[str]) -> pd.DataFrame:
    """
    return DataFrame with 2 columns:
        name: strings
        text: List(String)
    """
    df = pd.DataFrame(columns=["name", "text"])
    for file in files:
        df_tmp = pd.read_csv(file)[["name", "text"]]

        # Convert the 'text' column from string to list of strings
        df_tmp["text"] = df_tmp["text"].apply(lambda x: [x])

        # Group by 'name' and aggregate the lists of text
        df_tmp = df_tmp.groupby("name")["text"].sum().reset_index()

        # concat to df
        df = pd.concat([df, df_tmp])

        # Group by 'name' and aggregate the lists of text
        df = df.groupby("name")["text"].sum().reset_index()
    return df


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 2

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def create_model_word2vec(
    files: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    epochs: int = 10,
    tokenizer: Callable[[str], List[str]] = word_tokenize,
) -> Word2Vec:

    start = time.time()

    if df is None:
        df = files_to_df(files)

    # nltk.download('punkt')

    print("begin of Tokenisation")

    # Tokenisation
    if files is None:
        sentences = [tokenizer(text) for text in tqdm(df["text"].str.lower())]
    else:
        sentences = [tokenizer(text.lower()) for texts in df["text"] for text in texts]
    print(sum(len(s) for s in sentences))

    print("End of Tokenisation")

    print("begin train")
    # Creation of model Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=1,
    )

    print("Epoch #1 end")

    # train word2vec model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs - 1,
        callbacks=[EpochLogger()],
    )

    # how many time the train take
    print("Time to compute :", time.time() - start)

    return model
