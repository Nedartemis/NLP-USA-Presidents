from typing import Callable, Optional

import gensim.downloader
import keras
import numpy as np
import numpy.typing as npt
import tensorflow.python.keras.layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import torch.optim as optim
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegressionModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MultiLayerPerceptron2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        activation=F.relu,
    ):
        super(MultiLayerPerceptron2, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim * 2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        X = self.softmax(x)
        return x


TYPE_TOKENIZER = Callable[[str], npt.NDArray[np.str_]]
TYPE_WORD2VEC = Word2Vec
TYPE_POOLING_FUNCTION = Callable[[npt.NDArray], npt.NDArray]
TYPE_DOCUMENT_VECTORIZER = Callable[[str], npt.NDArray[np.float32]]


class MyFnnTorch:

    def __init__(
        self,
        num_classes: int,
        model: Optional[nn.Module] = None,
        optimizer=None,
        nb_epochs: int = 50,
        tokenizer: TYPE_TOKENIZER = word_tokenize,
        word2vec: Optional[KeyedVectors] = None,
        pooling_func: TYPE_POOLING_FUNCTION = lambda word_vectors: np.mean(
            word_vectors, axis=0
        ),
        count_vectorizer: Optional[CountVectorizer] = None,
        learning_rate: float = 0.05,
        activation=F.relu,
    ):

        if word2vec is not None:
            vocab_size = word2vec.vector_size
        elif count_vectorizer is not None:
            self.vectorize_X = lambda x: count_vectorizer.transform(x).toarray()
            vocab_size = len(count_vectorizer.vocabulary_)
        else:
            word2vec = word2vec or gensim.downloader.load("glove-twitter-50")

        if word2vec is not None:
            self.vectorize_X = self.create_vectorize_X(
                tokenizer, word2vec, pooling_func
            )

        self.model: nn.Module = model or MultiLayerPerceptron2(
            vocab_size=vocab_size,
            hidden_dim=150,
            num_classes=num_classes,
            activation=activation,
        )

        self.criterion = nn.CrossEntropyLoss()

        # optimizer = optim.SGD(model.parameters(), lr=0.1)
        # optimizer = optim.Adam(model.parameters(), lr=0.005)
        self.optimizer = optimizer or optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )

        self.nb_epochs = nb_epochs

    def create_vectorize_X(
        self,
        tokenizer: TYPE_TOKENIZER,
        word2vec: TYPE_WORD2VEC,
        pooling_func: TYPE_POOLING_FUNCTION,
    ):
        def document_vectorizer(doc: str) -> npt.NDArray[np.float32]:
            words = tokenizer(doc)
            word_vectors = np.array(
                [word2vec[word] for word in words if word in word2vec]
            )

            if len(word_vectors) == 0:
                return np.zeros(word2vec.vector_size)

            return pooling_func(word_vectors)

        def func(X_text: npt.NDArray[np.str_]):
            X = np.array([document_vectorizer(text) for text in X_text])
            return X

        return func

    def y_string_to_integer_encoded(
        self, y: npt.NDArray[np.str_]
    ) -> npt.NDArray[np.int32]:
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(
            y
        )  # Transform classes to integers
        return integer_encoded

    def transform_X(self, X_text: npt.NDArray[np.str_]) -> torch.Tensor:

        X = self.vectorize_X(X_text)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return X_tensor

    def transform_y(self, y_text: npt.NDArray[np.str_]) -> torch.Tensor:
        y = self.y_string_to_integer_encoded(y_text)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return y_tensor

    def fit(
        self,
        X_train_text: npt.NDArray[np.str_],
        y_train_text: npt.NDArray[np.str_],
        X_test_text: npt.NDArray[np.str_] = None,
        y_test_text: npt.NDArray[np.str_] = None,
        batch_size: int = 64,
    ):

        X_train = self.transform_X(X_train_text)
        y_train = self.transform_y(y_train_text)
        if X_test_text is not None:
            X_test = self.transform_X(X_test_text)
            y_test = self.transform_y(y_test_text)

        # Create TensorDataset objects
        train_dataset = TensorDataset(X_train, y_train)
        if X_test_text is not None:
            test_dataset = TensorDataset(X_test, y_test)

        # Create DataLoader objects
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        if X_test_text is not None:
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False
            )

        for epoch in range(self.nb_epochs):
            for inputs, labels in train_loader:
                inputs = inputs.float()
                labels = labels.long()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{self.nb_epochs}], Loss: {loss.item():.4f}")

            self.model.eval()

            if X_test_text is not None:
                all_labels = []
                all_predictions = []

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        all_labels.extend(labels)
                        all_predictions.extend(predicted)

                precision = precision_score(
                    all_labels, all_predictions, average="weighted", zero_division=0
                )
                recall = recall_score(all_labels, all_predictions, average="weighted")
                f1 = f1_score(all_labels, all_predictions, average="weighted")

                print(
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
                )

    def predict_proba(self, text: npt.NDArray[np.str_]) -> torch.Tensor:
        X_tensor = self.transform_X(text)
        outputs = self.model(X_tensor)
        return nnf.softmax(outputs, dim=1)

    def predict(self, text: npt.NDArray[np.str_]) -> str:
        probas = self.predict_proba(text)
        _, predicted = torch.max(probas, 1)
        return self.label_encoder.inverse_transform(predicted)
