import pandas
import os.path
from .utils import Log, plot_distribution, add_test_log
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def drug_main(data_path: str, result_path: str):
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

    nb_classifier = MultinomialNB()
    bdt_classifier = DecisionTreeClassifier()
    tdt_classifier = GridSearchCV(DecisionTreeClassifier(), {
        'criterion': ('gini', 'entropy'), 
        'max_depth': (5, 20), 
        'min_samples_split': (5, 20, 40)
    })
    per_classifier = Perceptron()
    bmlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
    tmlp_classifier = GridSearchCV(MLPClassifier(), {
        'activation': ('logistic', 'tanh', 'relu', 'identity'), 
        'hidden_layer_sizes': ((30, 50), (10, 10, 10)),
        'solver': ('adam', 'sgd')
    })

    classifiers = [nb_classifier, bdt_classifier, tdt_classifier, per_classifier, bmlp_classifier, tmlp_classifier]

    log = Log()

    for classifier in classifiers:
        classifier.fit(x_train, y_train)

        title = "*" * 10

        if hasattr(classifier, 'estimator'):
            title += " GridSearchCV(" + str(classifier.estimator).split("(")[0] + ")"
        else:
            title += " " + str(classifier).split("(")[0]

        if hasattr(classifier, 'best_params_'):
            title += ", " + str(classifier.best_params_)

        title += " " + "*" * 10

        log.label("(a)", title)

        y_test_predict = classifier.predict(x_test)

        add_test_log(log, classifier, x_test, y_test)

    log.save(os.path.join(result_path, "./drug-performance.txt"))
