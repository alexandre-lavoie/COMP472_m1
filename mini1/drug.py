import pandas
import math
import os.path
from .utils import Log, get_classifier_name, plot_distribution, add_test_log
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

import warnings

def drug_main(data_path: str, result_path: str):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    features = pandas.read_csv("./datasets/drug200.csv")
    labels = features.pop("Drug")

    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    label_counts = dict(label_counts)

    plot_distribution(class_counts=label_counts, result_path=result_path, label="Drug")

    features = pandas.get_dummies(features, columns=['Sex','BP','Cholesterol'])
    classes = pandas.Categorical(features)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8)

    log = Log()

    classifier_run_stats = []

    classifier_generators = [
        lambda: MultinomialNB(),
        lambda: DecisionTreeClassifier(),
        lambda: GridSearchCV(DecisionTreeClassifier(), {
            'criterion': ('gini', 'entropy'), 
            'max_depth': (5, 20), 
            'min_samples_split': (5, 20, 40)
        }),
        lambda: Perceptron(),
        lambda: MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd'),
        lambda: GridSearchCV(MLPClassifier(), {
            'activation': ('logistic', 'tanh', 'relu', 'identity'), 
            'hidden_layer_sizes': ((30, 50), (10, 10, 10)),
            'solver': ('adam', 'sgd')
        })
    ]

    RUN_COUNT = 10
    for run_id in range(1, RUN_COUNT + 1):
        print(f"Try {run_id}.")

        classifier_stats = []

        for classifier_generator in classifier_generators:
            classifier = classifier_generator()
            classifier.fit(x_train, y_train)

            title = "*" * 10
            title + " " + get_classifier_name(classifier)

            if hasattr(classifier, 'best_params_'):
                title += ", " + str(classifier.best_params_)

            title += f" - Try {run_id} "
            title += " " + "*" * 10

            log.label("(a)", title)

            y_test_predict = add_test_log(log, classifier, x_test, y_test)

            accuracy = accuracy_score(y_test, y_test_predict)
            macro_f1 = f1_score(y_test, y_test_predict, average="macro")
            weighted_f1 = f1_score(y_test, y_test_predict, average="weighted")

            classifier_stats.append((accuracy, macro_f1, weighted_f1))

        classifier_run_stats.append(classifier_stats)

    log.label("(8)", "*" * 10 + " Classifier Stats " + "*" * 10)

    for classifier_generator, classifier_stats in zip(classifier_generators, zip(*classifier_run_stats)):
        classifier = classifier_generator()

        log.add(f"{get_classifier_name(classifier)}:\n")

        for stat_index, label in zip(range(3), ["Accuracy", "Macro F1", "Weighted F1"]):
            stats = [cs[stat_index] for cs in classifier_stats]
            stat_average = sum(stats) / len(stats)
            stat_stdev = math.sqrt(sum((s - stat_average) ** 2 for s in stats) / len(stats))

            log.add(f"{label} Average: {stat_average}")
            log.add(f"{label} Stdev: {stat_stdev}")

        log.add("")

    log.save(os.path.join(result_path, "./drug-performance.txt"))

    print("\nDone!")
