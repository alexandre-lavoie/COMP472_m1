import zipfile
import os
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np
from .utils import Log
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Dataset = Dict[str, List[str]]

def extract_dataset(zip_path: str, data_path: str):
    zip_file = zipfile.ZipFile(zip_path)

    if not os.path.exists(os.path.join(data_path, "./BBC")):
        zip_file.extractall(data_path)

def load_dataset(data_path: str) -> Dataset:
    dataset = defaultdict(list)

    for file_path in glob.glob(os.path.join(data_path, "./BBC/") + "**/*"):
        if file_path.endswith("README.txt"): continue

        data_type = file_path.split("/")[-2]

        with open(file_path, "r", encoding="latin1") as h:
            data = h.read()

        dataset[data_type].append(data)

    return dataset

def plot_distribution(class_counts: dict, result_path: str):
    dataset_labels = list(class_counts.keys())
    dataset_counts = list(class_counts.values())

    plt.title("BBC Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")

    plt.bar(dataset_labels, dataset_counts)

    plt.savefig(os.path.join(result_path, "./BBC-distribution.pdf"))

def get_vectorizer(dataset: Dataset) -> CountVectorizer:
    corpus = []

    for data in dataset.values():
        corpus += data

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit_transform(corpus)

    return vectorizer

def parse_dataset(dataset: Dataset, vectorizer: CountVectorizer) -> Tuple[List[str], ]:
    lines = []
    y_dataset = []

    for label, texts in dataset.items():
        for text in texts:
            lines.append(text)
            y_dataset.append(label)

    x_dataset = vectorizer.transform(lines)

    return x_dataset, y_dataset

def add_test_log(log: Log, classifier: MultinomialNB, x_test: any, y_test: any):
    y_test_predict = classifier.predict(x_test)

    test_confusion_matrix = confusion_matrix(y_test, y_test_predict)
    log.label("b)", test_confusion_matrix)

    test_classification_report = classification_report(y_test, y_test_predict)
    log.label("c)", test_classification_report)

    test_accuracy_score = accuracy_score(y_test, y_test_predict)
    log.label("d)", test_accuracy_score)

def bbc_main():
    data_path="./data"

    if not os.path.exists(data_path): os.makedirs(data_path, exist_ok=True)
    
    result_path="./results"
    
    if not os.path.exists(result_path): os.makedirs(result_path, exist_ok=True)

    extract_dataset(
        zip_path="./datasets/BBC-20210914T194535Z-001.zip",
        data_path=data_path
    )

    dataset = load_dataset(
        data_path=data_path
    )

    class_counts = dict([(l, len(vs)) for l, vs in dataset.items()])

    plot_distribution(
        class_counts=class_counts,
        result_path=result_path
    )

    vectorizer = get_vectorizer(dataset)

    x_dataset, y_dataset = parse_dataset(
        dataset=dataset,
        vectorizer=vectorizer
    )

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.8)

    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    log = Log()

    add_test_log(
        log=log,
        classifier=classifier, 
        x_test=x_test,
        y_test=y_test
    )

    vocabulary = len(vectorizer.get_feature_names())

    class_words = {}
    i = 0
    for label, count in class_counts.items():
        class_words[label] = np.sum(x_dataset[i:count,:])
        i += count

