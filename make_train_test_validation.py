import datasets
import json
import os

def save_train_test_validation(dataset, name):
    x = dataset["train"].train_test_split(train_size=0.6, test_size=0.4, seed=42)
    y = x["test"].train_test_split(train_size=0.5, test_size=0.5, seed=42)

    x["train"].to_json(name + r"\train.json")
    y["train"].to_json(name + r"\validation.json")
    y["test"].to_json(name + r"\test.json")

os.chdir(r"C:\Projekter\Natural Language Processing\z\project-milestone-p2-group21\smoldata")

en_yue_sentences: datasets.dataset_dict.DatasetDict = datasets.load_dataset("google/smol", "smolsent__en_yue") # type: ignore
en_yue_docs: datasets.dataset_dict.DatasetDict = datasets.load_dataset("google/smol", "smoldoc__en_yue")       # type: ignore
en_yue_characters: datasets.dataset_dict.DatasetDict = datasets.load_dataset("google/smol", "gatitos__en_yue") # type: ignore

save_train_test_validation(en_yue_sentences, "smol_en_yue_sentences")
save_train_test_validation(en_yue_docs, "smol_en_yue_docs")
save_train_test_validation(en_yue_characters, "gatitos_yue_zh_characters")

