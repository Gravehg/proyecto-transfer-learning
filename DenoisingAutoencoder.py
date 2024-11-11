import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt
import torchvision

class DenoisingAutoEncoder(L.LightningModule):
    def __init__(self,input_size,latent_dim,lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1), #224x224 -> 112x112
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),#112x112 -> 56x56
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1), #56x56 -> 28x28
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1), #28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(512,latent_dim,kernel_size=4,stride=2,padding=1), #14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(), # Convierte 2D a 1D
            nn.Linear(latent_dim*7*7,latent_dim) # Reduce la dimension a latent_dim
        )
        
        self.decoder = nn.Sequential(
            nn.linear(latent_dim,latent_dim*7*7),
            nn.ReLU(),
            nn.Unflatten(1,(latent_dim,7,7)),
            nn.ConvTranspose2d(latent_dim,512,kernel_size=4,stride=2,padding=1), #7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1), #14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1), #28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1), #56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1), #112x112 -> 224x224
            nn.Sigmoid()
        )
        self.lr = lr
        self.reconstructed_images = []
        self.original_images = []

    def forward(self,x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer
    
    def add_noise(self,x,noise_level=0.1): # Salt and pepper noise
        noisy = x.clone()
        rnd = torch.rand(noisy.size())
        noisy[rnd<noise_level/2] = 0 # Salt noise
        noisy[rnd>1-noise_level/2] = 1 # Pepper noise
        return noisy
    
    def training_step(self,train_batch,batch_idx):
        x,_ = train_batch # Ignora las etiquetas
        x_noisy = self.add_noise(x) # Agrega ruido
        z = self.encoder(x_noisy) # Codifica la imagen ruidosa
        x_hat = self.decoder(z) # Decodifica la imagen ruidosa
        loss = F.mse_loss(x_hat,x) # Calcula la perdida
        self.log("train_loss",loss) # Guarda la perdida
        return loss

    def validation_step(self,val_batch,batch_idx):
        x,_ = val_batch
        x_noisy = self.add_noise(x)
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat,x)
        self.log("val_loss",loss)

        if batch_idx == 0:
            self.original_images = x[:8] # Guarda las primeras 8 imagenes
            self.reconstructed_images = x_hat[:8] # Guarda las primeras 8 imagenes reconstruidas

    def on_validation_epoch_end(self):
        grid_original = torchvision.utils.make_grid(self.original_images)# Crea una cuadricula con las imagenes originales
        grid_reconstructed = torchvision.utils.make_grid(self.reconstructed_images)
        self.logger.experiment.add_image("original",grid_original,self.current_epoch)
        self.logger.experiment.add_image("reconstructed",grid_reconstructed,self.current_epoch)

    

        