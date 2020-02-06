import sys
import numpy as np

import torch
from torch import nn
from torch.nn.utils import weight_norm

import bee_view_analyzer.Utils as Utils

class VAE(nn.Module):
    def __init__(self, num_hidden=128, num_latents=2):
        super(VAE, self).__init__()
        
        self.num_latents = num_latents
        self.num_hidden = num_hidden
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_hidden // 16, 3, padding=1),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 16, num_hidden // 16, 3, stride=2, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 16, num_hidden // 8, 3, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 8, num_hidden // 8, 3, stride=2, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 8, num_hidden // 4, 3, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 4, num_hidden // 4, 3, stride=2, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 4, num_hidden // 2, 3, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 2, num_hidden // 2, 3, stride=2, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 2, num_hidden // 1, 3, padding=1)),
            nn.SELU(),
        )
        
        self.to_latent = nn.Linear(num_hidden * 4 * 4, num_latents * 2)
        self.from_latent = nn.Linear(num_latents, num_hidden * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(num_hidden // 1, num_hidden // 1, 3, padding=1),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 1, num_hidden // 2, 3, padding=1)),
            nn.SELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            weight_norm(nn.Conv2d(num_hidden // 2, num_hidden // 2, 3, padding=1)),
            nn.SELU(),
            weight_norm(nn.Conv2d(num_hidden // 2, num_hidden // 4, 3, padding=1)),
            nn.SELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            weight_norm(nn.Conv2d(num_hidden // 4, num_hidden // 4, 3, padding=1)),
            nn.ReLU(),
            weight_norm(nn.Conv2d(num_hidden // 4, num_hidden // 8, 3, padding=1)),
            nn.SELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            weight_norm(nn.Conv2d(num_hidden // 8, num_hidden // 8, 3, padding=1)),
            nn.ReLU(),
            weight_norm(nn.Conv2d(num_hidden // 8, num_hidden // 16, 3, padding=1)),
            nn.SELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            weight_norm(nn.Conv2d(num_hidden // 16, num_hidden // 16, 3, padding=1)),
            nn.SELU(),
            nn.Conv2d(num_hidden // 16, 3, 3, padding=1)
        )

    def encode(self, x):
        z_conv = self.encoder(x)
        z_flat = z_conv.view(-1, self.num_hidden * 4 * 4)
        z = self.to_latent(z_flat)
        
        z_mu = z[:, :self.num_latents]
        z_logvar = z[:, self.num_latents:]
        
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z_flat = self.from_latent(z)
        z_conv = z_flat.view(-1, self.num_hidden, 4, 4)
        x_logits = self.decoder(z_conv)
        return torch.sigmoid(x_logits), x_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat, xhat_logits = self.decode(z)
        xhat = xhat[:, :, :, 4:-5]
        xhat_logits = xhat_logits[:, :, :, 4:-5]
        
        return xhat, xhat_logits, mu, logvar

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(vae_num_latents, vae_num_hidden, path):
    model = VAE(num_latents=vae_num_latents, num_hidden=vae_num_hidden)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
                
    model.eval()
    
    return model

def init_weights(model):
    _ = model.apply(_init_weights)
    
    return model

def train_model(model, num_epochs, batch_size, optimizer, train_loader, valid_loader, h, w, device, mask, save_path):
    losses = []
    val_losses = []
    
    i = 0
    fade_in_max = 10000
    
    for i_epoch in range(num_epochs):
        val_loss = []
        model.eval()
        
        with torch.no_grad():
            for i_batch, (xb, yb) in enumerate(valid_loader):
                #xb = xb.mean(dim=1, keepdim=True)
                #xb = xb.view(batch_size, -1).to(device)
                xb = xb.to(device)
                xb_hat, xb_hat_logits, z_mu, z_logvar = model(xb)
                beta = min((1, i / fade_in_max)) / 10
                
                loss = _loss_function(h, w, xb_hat_logits.contiguous().view(batch_size, -1), 
                                 xb.view(batch_size, -1), z_mu, z_logvar, mask, beta)
                                 
                val_loss.append(loss.cpu().item())
        val_losses.append(np.mean(val_loss))
            
        for i_batch, (xb, yb) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            #xb = xb.mean(dim=1, keepdim=True)
            #xb = xb.view(batch_size, -1).to(device)
            xb = xb.to(device)
            xb_hat, xb_hat_logits, z_mu, z_logvar = model(xb)
        
            i += 1
            beta = min((1, i / fade_in_max)) / 10
        
            loss = _loss_function(h, w, xb_hat_logits.contiguous().view(batch_size, -1), 
                                 xb.view(batch_size, -1), z_mu, z_logvar, mask, beta)
        
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().item())
        
            sys.stdout.write('\r{}_{} - train:{:.2f}, val:{:.2f}, beta:{:.2f}'.format(
                i_epoch, i_batch, np.median(losses[-1000:]), val_losses[-1], beta))
                
        # save intermediate model after each epoch in case training is cancelled before all epochs are finished
        save_model(model, save_path)
        Utils.save_object(save_path + "losses.dill", losses)
        Utils.save_object(save_path + "val_losses.dill", val_losses)
        
    return model, losses, val_losses
    
    
def _loss_function(h, w, recon_x, x, mu, logvar, mask, beta=1):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
    bce = bce.view(-1, 3, h, w)    
    bce *= mask.unsqueeze(0).unsqueeze(0)
    bce = bce.sum(dim=(1, 2, 3)).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    
    return bce + beta * KLD
    
def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, 0.0, np.sqrt(1/(3*3)))
        m.bias.data.fill_(0.01)
        
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)
