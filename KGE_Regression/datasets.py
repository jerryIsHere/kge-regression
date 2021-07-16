import torch

max_token_length = 512
from sklearn.model_selection import train_test_split


def build_train_test_dataset(
    embedding_smile_path, tokenizer, train_size=0.8, test_size=0.2
):
    embedding_smile = pd.read_pickle(embedding_smile_path)
    train, test = train_test_split(
        embedding_smile, train_size=train_size, test_size=test_size, random_state=42
    )
    return (train_dataset(train, tokenizer), test_dataset(test, tokenizer))


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, train, tokenizer, start=0):
        super(train_dataset).__init__()
        assert len(train) > start, "this example code only works with end >= start"
        self.start = start
        self.train = train
        self.end = len(train)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train)

    def __getitem__(self, key):
        token = self.tokenizer.encode(
            text=str(self.train.iloc[key]["smile"]),
            padding="max_length",
            max_length=max_token_length,
            truncation=True,
        )

        return {  # labels: torch.tensor(self.labels[key]),
            "input_ids": torch.tensor(token),
            "labels": torch.tensor(self.train.iloc[key]["embedding"]),
        }


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, test, tokenizer, start=0):
        super(test_dataset).__init__()
        assert len(test) > start, "this example code only works with end >= start"
        self.start = start
        self.end = len(test)
        self.test = test
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.test)

    def __getitem__(self, key):
        token = self.tokenizer.encode(
            text=str(self.test.iloc[key]["smile"]),
            padding="max_length",
            max_length=max_token_length,
            truncation=True,
        )

        return {  #'labels': torch.tensor(self.labels[key]),
            "input_ids": torch.tensor(token),
            "labels": torch.tensor(self.test.iloc[key]["embedding"]),
        }


import pandas as pd


class inference_dataset(torch.utils.data.Dataset):
    def __init__(self, masked_cid_smile_path, tokenizer, start=0):
        super(inference_dataset).__init__()
        masked_cid_smile = pd.read_pickle(masked_cid_smile_path)
        max_token_length = 512
        assert (
            len(masked_cid_smile) > start
        ), "this example code only works with end >= start"
        self.start = start
        self.pd = masked_cid_smile
        self.end = len(masked_cid_smile)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pd)

    def __getitem__(self, key):
        token = self.tokenizer.encode(
            text=str(self.pd.iloc[key]["smile"]),
            padding="max_length",
            max_length=max_token_length,
            truncation=True,
        )

        return {  # labels: torch.tensor(self.labels[key]),
            "input_ids": torch.tensor(token)
        }


import numpy as np


def evaluate_fn(eval):
    logits, labels = (eval.predictions, eval.label_ids)
    loss = np.mean(np.absolute(np.squeeze(logits) - labels))
    return {"eval_loss": loss}
