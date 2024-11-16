import torch.nn as nn
import torch.optim
import pytorch_lightning as pl

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
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_function(x_hat, x)
        self.log("cl_test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_end(self):
        # Save the encoder weights
        torch.save(self.state_dict(), self.encoder_path)
        print(f"Encoder weights saved to {self.encoder_path}")



