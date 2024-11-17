
import os
from torch.utils.data import DataLoader,Subset, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl


class ButterflyDataModule(pl.LightningDataModule):
    def __init__ (self, data_dir, batch_size,needs_split,split_ratio,num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.needs_split  = needs_split
        self.split_ratio= split_ratio

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'valid'), transform=transform)
        self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=transform)
        if self.needs_split:
            unsupervised_size = int(self.split_ratio * len(self.train_dataset))
            supervised_size = len(self.train_dataset) - unsupervised_size
            print(unsupervised_size, supervised_size)
            self.unsupervised_train_dataset, self.supervised_train_dataset = random_split(
                self.train_dataset, [unsupervised_size, supervised_size]
            )
    
    def unsupervised_train_loader(self):
        return DataLoader(self.unsupervised_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)
    
    def supervised_train_loader(self):
        return DataLoader(self.supervised_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    #USAR PARA HACER LA VALIDACION EN EL UNET
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)