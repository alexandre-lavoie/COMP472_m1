import zipfile
import os
import os.path
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import math
from .utils import Log, plot_distribution, add_test_log
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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

def get_vectorizer(dataset: Dataset) -> CountVectorizer:
    corpus = []

    for data in dataset.values():
        corpus += data

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit_transform(corpus)

    return vectorizer

def parse_dataset(dataset: Dataset, vectorizer: CountVectorizer) -> Tuple[csr_matrix, List[str]]:
    lines = []
    y_dataset = []

    for label, texts in dataset.items():
        for text in texts:
            lines.append(text)
            y_dataset.append(label)

    x_dataset = vectorizer.transform(lines)

    return x_dataset, y_dataset

def add_stats_log(log: Log, classifier: MultinomialNB, dataset: dict, vectorizer: CountVectorizer, favorite_words: List[str], x_dataset: csr_matrix, y_dataset: List[str]):
    prior_probabilities = dict(zip(dataset.keys(), (math.exp(lp) for lp in classifier.class_log_prior_)))

    log.label("(e)", json.dumps(prior_probabilities, indent=4))

    vocabulary = len(vectorizer.get_feature_names())

    log.label("(f)", vocabulary)

    class_words = defaultdict(int)
    for i, label in enumerate(y_dataset):
        class_words[label] += np.sum(x_dataset[i,:])
    class_words = dict((label, int(val)) for label, val in class_words.items())

    log.label("(g)", json.dumps(class_words, indent=4))

    corpus_words = sum(class_words.values())

    log.label("(h)", corpus_words)

    zero_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for i, label in enumerate(y_dataset):
        zero_count = vocabulary - len(x_dataset[i,:].nonzero()[0])

        zero_counts[label] += zero_count
        total_counts[label] += vocabulary

    zero_totals = dict(zip(zero_counts.keys(), zip(zero_counts.values(), total_counts.values())))
    zero_totals_text = json.dumps(dict((key, f"{zero} / {total} ({(zero / total) * 100}%)") for key, (zero, total) in zero_totals.items()), indent=4)

    log.label("(i)", zero_totals_text)

    zero_totals_corpus = sum(v1 for (v1, v2) in zero_totals.values())
    totals_corpus =  sum(v2 for (v1, v2) in zero_totals.values())

    zero_corpus_text = f"{zero_totals_corpus} / {totals_corpus} ({(zero_totals_corpus / totals_corpus) * 100}%)"

    log.label("(j)", zero_corpus_text)

    favorite_word_indices = vectorizer.transform(favorite_words).nonzero()[1]
    
    probs_text = ""
    for favorite_word, favorite_word_index in zip(favorite_words, favorite_word_indices):
        probs_text += f"{favorite_word}:\n"

        label_probs = {}
        for label, features_log_prob in zip(dataset.keys(), classifier.feature_log_prob_):
            feature_log_prob = features_log_prob[favorite_word_index]
            label_probs[label] = feature_log_prob

        probs_text += json.dumps(label_probs, indent=4) + "\n\n"

    log.label("(k)", probs_text.strip())

def build_classifier(x_train, y_train, smoothing: float = 1.0) -> MultinomialNB:
    classifier = MultinomialNB(alpha=smoothing)
    classifier.fit(x_train, y_train)

    return classifier

def perform_test(log: Log, title: str, vectorizer: CountVectorizer, x_train, x_test, y_train, y_test, x_dataset, y_dataset, dataset, favorite_words, smoothing: float = 1.0):
    classifier = build_classifier(x_train, y_train, smoothing)

    log.label("(a)", title)

    add_test_log(
        log=log,
        classifier=classifier, 
        x_test=x_test,
        y_test=y_test
    )

    add_stats_log(
        log=log,
        dataset=dataset,
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        vectorizer=vectorizer,
        classifier=classifier,
        favorite_words=favorite_words,
    )

def bbc_main(data_path: str, result_path: str):
    extract_dataset(
        zip_path="./datasets/BBC-20210914T194535Z-001.zip",
        data_path=data_path
    )

    dataset = load_dataset(
        data_path=data_path
    )

    class_counts = dict([(l, len(vs)) for l, vs in dataset.items()])

    plot_distribution(
        label="BBC",
        class_counts=class_counts,
        result_path=result_path
    )

    vectorizer = get_vectorizer(dataset)

    x_dataset, y_dataset = parse_dataset(
        dataset=dataset,
        vectorizer=vectorizer
    )

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.8)

    log = Log()

    for title, smoothing in [("Default, Try 1", 1.0), ("Default, Try 2", 1.0), ("Smoothing 0.0001", 0.0001), ("Smoothing 0.9", 0.9)]:
        print(title)

        perform_test(
            title="*" * 10 + " MultinomialNB, " + title + " " + "*" * 10,
            favorite_words=["Google", "French"],
            log=log,
            smoothing=smoothing,
            vectorizer=vectorizer,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dataset=dataset,
            x_dataset=x_dataset,
            y_dataset=y_dataset
        )

    log.save(os.path.join(result_path, "./bbc-performance.txt"))

    print("\nDone!")
