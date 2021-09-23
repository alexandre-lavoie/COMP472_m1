import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os.path

class Log:
    __log: str

    def __init__(self):
        self.__log = ""

    def add(self, line: str):
        self.__log += str(line) + "\n"

    def label(self, label: str, line: str):
        self.add(label + "\n\n" + str(line) + "\n")

    def save(self, path: str):
        with open(path, "w") as h:
            h.write(str(self))

    def __str__(self) -> str:
        return self.__log.strip()

def plot_distribution(class_counts: dict, result_path: str, label: str):
    dataset_labels = list(class_counts.keys())
    dataset_counts = list(class_counts.values())

    plt.title(f"{label} Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.bar(dataset_labels, dataset_counts)

    plt.savefig(os.path.join(result_path, f"./{label.lower()}-distribution.pdf"))

def add_test_log(log: Log, classifier: any, x_test: any, y_test: any):
    y_test_predict = classifier.predict(x_test)

    test_confusion_matrix = confusion_matrix(y_test, y_test_predict)

    log.label("(b)", test_confusion_matrix)

    test_classification_report = classification_report(y_test, y_test_predict)

    log.label("(c)", test_classification_report)

    test_accuracy_score = accuracy_score(y_test, y_test_predict)

    log.label("(d)", test_accuracy_score)
