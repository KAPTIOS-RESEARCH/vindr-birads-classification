from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import os
import torch
from google.cloud import storage
from src.data.sets.vindr_remote import VindrMammoDataset

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"

client = storage.Client()
bucket_name = 'vindr-mammo-dataset'
bucket = client.bucket(bucket_name)


class VindrRemoteDataloader(object):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 debug: bool = True):

        super(VindrRemoteDataloader, self).__init__()
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train(self):
        train_dataset = VindrMammoDataset(
            bucket,
            split='training'
        )
        class_counts = torch.bincount(torch.tensor(
            train_dataset.labels, dtype=torch.long))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_dataset.labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                sampler=sampler,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = VindrMammoDataset(
            bucket,
            split='test'
        )

        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=False,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()
