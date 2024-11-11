
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl


class ButterflyDataModule(pl.LightningDataModule):
    def __init__ (self, data_dir, batch_size,num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'valid'), transform=transform)
        self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)