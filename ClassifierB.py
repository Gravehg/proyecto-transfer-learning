import torch.nn as nn
import torch.optim
import pytorch_lightning as pl
import torchmetrics
import numpy as np

class Classifier(pl.LightningModule):
    def __init__(self, encoder, lr, num_classes, latent_dim, isFreezed, encoder_path):
        super(Classifier, self).__init__()
        self.encoder = encoder
        if isFreezed:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.encoder.train()
        self.fc1 = nn.Linear(latent_dim * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()  # Define ReLU layer
        self.lr = lr
        self.loss_function = nn.CrossEntropyLoss()
        self.encoder_path = encoder_path
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.droput1 = nn.Dropout(0.15)
        self.test_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    def forward(self, x):
        with torch.no_grad() if not self.training else torch.enable_grad():
            enc_output, _ = self.encoder(x)
        x = enc_output.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.droput1(x)
        x = self.fc2(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.droput1(x)
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

        # Store metrics
        self.test_metrics["loss"].append(loss.item())
        self.test_metrics["accuracy"].append(acc.item())
        self.test_metrics["precision"].append(precision.item())
        self.test_metrics["recall"].append(recall.item())
        self.test_metrics["f1"].append(f1.item())

        # Log individual batch metrics
        self.log("cl_test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("cl_test_accuracy", acc, on_step=True, prog_bar=True, logger=True)
        self.log("cl_test_precision", precision, on_step=True, prog_bar=True, logger=True)
        self.log("cl_test_recall", recall, on_step=True, prog_bar=True, logger=True)
        self.log("cl_test_f1", f1, on_step=True, prog_bar=True, logger=True)

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
        torch.save(self.state_dict(), self.encoder_path)
        print(f"Encoder weights saved to {self.encoder_path}")


