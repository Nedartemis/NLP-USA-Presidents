from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class MyLogisticRegression():
    def __init__(self, cv, clf: LogisticRegression):
        self.cv = cv
        self.clf = clf
        self.model = make_pipeline(self.cv, clf)

        self.fit = self.model.fit 
        self.predict = self.model.predict