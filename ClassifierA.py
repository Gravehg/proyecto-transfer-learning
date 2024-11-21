import torch.nn as nn
import torch.optim
import pytorch_lightning as pl
import torchmetrics
import numpy as np

class ClassifierFromZero(pl.LightningModule):
    def __init__(self, lr, num_classes, latent_dim,model_path):
        super(ClassifierFromZero,self).__init__()
        self.lr = lr
        self.latent_dim = latent_dim
        self.model_path = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256,latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()  # Define ReLU layer
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(latent_dim * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    def forward(self,x):
        x = self.conv1(x) #224x224x64
        x = self.relu(x)
        x = self.max_pool(x) #112x112x64
        x = self.conv2(x) #112x112x128
        x = self.relu(x)
        x = self.max_pool(x) #56x56x128
        x = self.conv3(x) #56x56x256
        x = self.relu(x)
        x = self.max_pool(x) #28x28x256
        x = self.conv4(x) #28x28x512
        x = self.relu(x)
        x = x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("cl_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("cl_val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        acc = self.accuracy(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1 = self.f1_score(y_hat, y)

        self.test_metrics["loss"].append(loss.item())
        self.test_metrics["accuracy"].append(acc.item())
        self.test_metrics["precision"].append(precision.item())
        self.test_metrics["recall"].append(recall.item())
        self.test_metrics["f1"].append(f1.item())


        self.log("cl_test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_precision", precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_test_epoch_end(self):
        # Calculate averages
        avg_loss = np.mean(self.test_metrics["loss"])
        avg_accuracy = np.mean(self.test_metrics["accuracy"])
        avg_precision = np.mean(self.test_metrics["precision"])
        avg_recall = np.mean(self.test_metrics["recall"])
        avg_f1 = np.mean(self.test_metrics["f1"])


        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True)
        self.log("avg_test_accuracy", avg_accuracy, prog_bar=True, logger=True)
        self.log("avg_test_precision", avg_precision, prog_bar=True, logger=True)
        self.log("avg_test_recall", avg_recall, prog_bar=True, logger=True)
        self.log("avg_test_f1", avg_f1, prog_bar=True, logger=True)

        self.test_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_end(self):
        # Save the encoder weights
        torch.save(self.state_dict(), self.model_path)
        print(f"Model weights saved to {self.model_path}")
