import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from classifier import Classifier


class HogPcaKnnClassifier(Classifier):
    def __init__(self):
        self.pca = PCA(n_components=7)
        self.knn = KNeighborsClassifier(7)  # odd number

    def fit(self, X, y):
        X_transformed = self.pca.fit_transform(X)
        self.knn.fit(X_transformed, y)

    def predict(self, X_test):
        X_test_transformed = self.pca.transform(X_test)
        y_pred = self.knn.predict(X_test_transformed)
        y_prob = self.knn.predict_proba(X_test_transformed)
        y_scores = np.array([max(y_prob[i]) for i in range(y_prob.shape[0])])
        return y_pred, y_prob, y_scores

