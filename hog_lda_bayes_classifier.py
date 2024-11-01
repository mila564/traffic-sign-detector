import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classifier import Classifier


class HogLdaBayesClassifier(Classifier):
    def __init__(self):
        self.lda = []
        for i in range(6):
            self.lda.append(LinearDiscriminantAnalysis())

    def fit(self, X, y):
        for i in range(len(X)):
            self.lda[i].fit_transform(X[i], y[i])

    def predict(self, X_test):
        y_pred = []
        y_prob = []

        for i in range(6):
            y_pred.append(self.lda[i].predict(X_test))
            y_prob.append(self.lda[i].predict_proba(X_test))

        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        final_pred = []
        final_scores = []

        for j in range(y_prob.shape[1]):
            # Get the positive probabilities
            column_scores = np.array([y_prob[:, j][i][1] for i in range(6)])
            column_pred = y_pred[:, j]
            index_max = np.argmax(column_scores)
            score = column_scores[index_max]
            final_scores.append(score)
            if score > 0.5:
                final_pred.append(column_pred[index_max])
            else:
                final_pred.append(0)
        return final_pred, y_prob, final_scores  # y_pred, y_prob, y_scores
