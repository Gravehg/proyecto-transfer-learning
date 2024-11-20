import os
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from sklearn.utils import shuffle
from torchvision import transforms, datasets
import numpy as np
from collections import defaultdict

class ButterflyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, needs_split, split_ratio, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.needs_split = needs_split
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
       
        # Load the datasets
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transforms)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'valid'), transform=train_transforms)
        self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=transform)

        # Perform stratified split if needed
        if self.needs_split:
            self.unsupervised_train_dataset, self.supervised_train_dataset = self.manual_stratified_split(self.train_dataset)

    def manual_stratified_split(self, dataset):
        labels = [label for _, label in dataset]
        label_to_indices = defaultdict(list)
        
        # Group indices by label
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        
        # Shuffle each label's indices and split
        np.random.seed(42)
        train_indices, test_indices = [], []
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)
            split_point = int(len(indices) * self.split_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
        
        train_indices, test_indices = shuffle(train_indices, random_state=42), shuffle(test_indices, random_state=42)
        
        unsupervised_train_dataset = Subset(dataset, train_indices)
        supervised_train_dataset = Subset(dataset, test_indices)

        return unsupervised_train_dataset, supervised_train_dataset

    def unsupervised_train_loader(self):
        return DataLoader(self.unsupervised_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
   
    def supervised_train_loader(self):
        return DataLoader(self.supervised_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)