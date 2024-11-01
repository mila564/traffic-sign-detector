import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report


class PerformanceEvaluator:

    # Confusion matrix
    @staticmethod
    def multiclass_confusion_matrix(y_val, y_pred):
        """
        Parameters:
            param y_pred: array-like of shape (n_samples,) Estimated targets as returned by a classifier.
            param y_val: array-like of shape (n_samples,) Ground truth (correct) target values.
        """

        cm = confusion_matrix(y_val, y_pred)
        cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1', '2', '3', '4', '5', '6'])
        cmd.plot()
        plt.show()
        return cmd.confusion_matrix

    # Code from https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    @staticmethod
    def plot_classification_report(cr, title='Classification report ', with_avg_total=False,
                                   cmap=plt.get_cmap("Blues")):
        lines = cr.split('\n')

        classes = []
        plotMat = []
        t = []

        for line in lines[2: (len(lines) - 5)]:
            t = line.split()
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            plotMat.append(v)

        if with_avg_total:
            aveTotal = lines[len(lines) - 1].split()
            classes.append('avg/total')
            vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
            plotMat.append(vAveTotal)

        plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(3)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.show()

    # Summarizing performance metrics
    def performance_metrics_report(self, y_val, y_pred):
        report = classification_report(y_val, y_pred, zero_division=0)
        print(report)
        self.plot_classification_report(report)
