import torch.nn as nn
import torch.optim
import pytorch_lightning as pl
import torchmetrics

class Classifier(pl.LightningModule):
    def __init__(self, encoder, lr, num_classes, latent_dim, isFreezed, encoder_path):
        super(Classifier, self).__init__()
        self.encoder = encoder
        if isFreezed:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
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

    def forward(self, x):
        with torch.no_grad() if not self.training else torch.enable_grad():
            enc_output, _ = self.encoder(x)
        x = enc_output.view(x.size(0), -1)
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

       
        self.log("cl_test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_precision", precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cl_test_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_end(self):
        # Save the encoder weights
        torch.save(self.state_dict(), self.encoder_path)
        print(f"Encoder weights saved to {self.encoder_path}")



