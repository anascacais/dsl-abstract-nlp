import logging

logging.disable(logging.INFO)
logging.disable(logging.WARNING)

import pickle
import json
import os
from pathlib import Path
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast


class ArxivDataset(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        batch_size=1,
        test_batch_size=1,
        path="/mnt/dataset/arxiv-dataset/arxiv-dataset",
    ):
        super().__init__()
        self.data_dir = path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset = None

    @classmethod
    def from_pickles(
        cls,
        tokenizer,
        train_path,
        val_path,
        test_path,
        batch_size=1,
        test_batch_size=1,
    ):
        assert (
            Path(train_path).is_file()
            and Path(val_path).is_file()
            and Path(test_path).is_file()
        )
        d = cls(
            tokenizer=tokenizer,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            path=None,
        )
        d.dataset = {"train": [], "val": [], "test": []}

        with open(train_path, "rb") as handle:
            d.dataset["train"] = pickle.load(handle)

        with open(val_path, "rb") as handle:
            d.dataset["val"] = pickle.load(handle)

        with open(test_path, "rb") as handle:
            d.dataset["test"] = pickle.load(handle)

        return d

    def prepare_data(self):
        assert not self.dataset

        # Check if data_dir is a directory
        dirpath = Path(self.data_dir)
        assert dirpath.is_dir()

        print(self.tokenizer.vocab_size)
        # Check if dataset is already processed
        self.dataset = {"train": [], "val": [], "test": []}
        for mode in self.dataset.keys():
            dataset_filename = dirpath / f"{mode}.txt"

            # If dataset file exists
            if dataset_filename.is_file():
                with tqdm(total=os.path.getsize(dataset_filename)) as pbar:
                    with open(str(dataset_filename), "r") as f:
                        while True:
                            # Get next line from file
                            line = f.readline()
                            if not line:
                                break
                            pbar.update(len(line))

                            entry = json.loads(line)
                            text = " ".join(entry["article_text"])
                            abstract = " ".join(entry["abstract_text"])
                            text_tokenized = torch.tensor(
                                self.tokenizer(text)["input_ids"], dtype=torch.int16
                            )
                            abstract_tokenized = torch.tensor(
                                self.tokenizer(abstract)["input_ids"],
                                dtype=torch.int16,
                            )
                            self.dataset[mode].append(
                                {
                                    "input_ids": text_tokenized,
                                    "output_ids": abstract_tokenized,
                                }
                            )

    def save_data(self, output_path="./", append=None):
        dirpath = Path(output_path)
        assert dirpath.is_dir()
        if append:
            append_str = f"-{append}"
        else:
            append_str = ""

        for mode in self.dataset.keys():
            with open(
                os.path.join(dirpath, f"{mode}{append_str}.pickle"), "wb"
            ) as handle:
                pickle.dump(
                    self.dataset[mode], handle, protocol=pickle.HIGHEST_PROTOCOL
                )

    def train_dataloader(self):
        assert len(self.dataset["train"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["train"]),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        assert len(self.dataset["val"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["val"]),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        assert len(self.dataset["test"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["test"]),
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def __str__(self):
        return str(self.dataset)


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]["input_ids"], self.dataset[idx]["output_ids"]

    def __len__(self):
        return len(self.dataset)


tokenizer = T5TokenizerFast.from_pretrained("t5-small")

dataset = ArxivDataset(tokenizer, batch_size=1, test_batch_size=1)
dataset.prepare_data()
dataset.save_data()

dataset_loaded = ArxivDataset.from_pickles(
    tokenizer=tokenizer,
    train_path="train.pickle",
    val_path="val.pickle",
    test_path="test.pickle",
)
