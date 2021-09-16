import zipfile
import os
import os.path
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
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

def plot_distribution(dataset: Dataset, result_path: str):
    dataset_labels = [label for label in dataset.keys()]
    dataset_counts = [len(data) for data in dataset.values()]

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

def parse_dataset(dataset: Dataset, vectorizer: CountVectorizer):
    parsed_dataset = []

    for label, texts in dataset.items():
        for text in texts:
            parsed_dataset.append((label, vectorizer.transform([text])))

    return parsed_dataset

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

    plot_distribution(
        dataset=dataset,
        result_path=result_path
    )

    vectorizer = get_vectorizer(dataset)

    parsed_dataset = parse_dataset(
        dataset=dataset,
        vectorizer=vectorizer
    )

    print(parsed_dataset[:2])
