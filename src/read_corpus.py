import os
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

import corpus.resources


def __preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # rename
    df.replace("Donald J. Trump", "Donald Trump", inplace=True)
    df.replace("Joseph R. Biden", "Joe Biden", inplace=True)

    return df


def read_corpus(
    categories: Optional[List[str]] = None,
    presidents: Optional[List[str]] = None,
    preprocess: bool = True,
) -> pd.DataFrame:
    """Read the corpus and store it into a dataframe.

    Args:
        categories (Optional[List[str]], optional): Filter on categories by keeping the given one. Defaults to None.
        presidents (Optional[List[str]], optional): Filter on presidents by keeping the given one. Defaults to None.

    Returns:
        pd.DataFrame: Represent the corpus, columns are category,name,date,text.
    """

    dir_resources = corpus.resources.__path__[0]
    files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dir_resources)
        for file in files
        if file.endswith(".csv")
    ]
    df = pd.concat([pd.read_csv(file) for file in tqdm(files)])

    if preprocess:
        df = __preprocess(df)
    if categories:
        df = df[df["category"].isin(categories)]
    if presidents:
        df = df[df["name"].isin(presidents)]

    return df


if __name__ == "__main__":
    df = read_corpus(categories=["debate", "speech"])
    print(df)
