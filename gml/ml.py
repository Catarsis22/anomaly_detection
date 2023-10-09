from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:
    def __init__(self):
        pass

    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    def f1_score(self, y_test, y_pred):
        return f1_score(y_test, y_pred)

