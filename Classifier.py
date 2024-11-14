import pytorch_lightning as pl
import torch.nn as nn
import torch.optim

class Classifier(pl.LightningModule):
    def __init__(Classifier,self, encoder,lr, num_classes, latent_dim, isFreezed):
        super(Classifier,self).__init__()
        self.encoder = encoder
        if isFreezed:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(latent_dim *28 *28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.lr = lr
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self,x):
        enc_output, _ = self.encoder(x)
        x = enc_output.view(x.size(0), -1)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = nn.ReLU(self.fc3(x))
        x = self.fc4(x)
        return x
        

    def training_step(self,batch,batch_idx):
        x,y =  batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("cl_train_loss",loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat,x)
        self.log("cl_test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    


