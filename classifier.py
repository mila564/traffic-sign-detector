from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
