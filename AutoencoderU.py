import torch
import torch.nn as nn
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.enc1 = self.conv_block(3,64)
        self.enc2 = self.conv_block(64,128)
        self.enc3 = self.conv_block(128,256)
        self.enc4 = self.conv_block(256, latent_dim)
    
    def conv_block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, padding =1),
            nn.ReLU()
        )
    
    def forward(self,x):
        enc1 = self.enc1(x) #224x224x64
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1)) #112x112x128
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2)) #56x56x256
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3)) #28x28x512
        return enc4, [enc1, enc2, enc3]
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(latent_dim, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)  
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.dec3 = self.conv_block(256, 128)  
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())
    
    def forward(self, x, skips):
        dec4 = self.upconv4(x)  #256x56x56
        dec4 = torch.cat([dec4, skips[2]], 1)   #512x56x56
        dec4 = self.dec4(dec4)  #256x56x56
        dec3 = self.upconv3(dec4)   #128x112x112
        dec3 = torch.cat((dec3, skips[1]), dim=1)  #256x112x112
        dec3 = self.dec3(dec3) #128x112x112
        dec2 = self.upconv2(dec3)  #64x224x224
        dec2 = torch.cat((dec2, skips[0]), dim=1) #128x224x224
        dec2 = self.dec2(dec2) #64x112x112
        return self.final_conv(dec2)


    

class AutoEncoder(pl.LightningModule):
    def __init__(self, lr, latent_dim):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.lr = lr
        self.loss_fn = nn.MSELoss()
    
    def forward(self,x):
        enc_output, skips = self.encoder(x)
        return self.decoder(enc_output,skips)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat,x)
        self.log("autoencoder_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        x, _  = batch
        x_hat = self(x)
        loss = self.loss_fn(x,x_hat)
        self.log("autoencoder_val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat,x)
        self.log("autoencoder_test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    

    def on_train_end(self):
        # Save the encoder weights
        encoder_path = "encoder_weights.pth"
        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder weights saved to {encoder_path}")
        