import torch
import torch.nn as nn
import torch.nn.functional as F

'''
这是一个简化版的 VQ-GAN, 去掉了过于复杂的 Attention 模块（对于简单几何体，纯卷积足够且更快），保留了 PatchGAN 判别器。
'''
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return x + self.block(x)

class VQGAN(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=64, num_embeddings=256):
        super().__init__()
        # Encoder: 256x256 -> 16x16 (Downsample factor 16 = 2^4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), # 128
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.Conv2d(256, embedding_dim, 4, 2, 1), # 16
            ResBlock(embedding_dim, embedding_dim),
            ResBlock(embedding_dim, embedding_dim)
        )
        
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder: 16x16 -> 256x256
        self.decoder = nn.Sequential(
            ResBlock(embedding_dim, embedding_dim),
            ResBlock(embedding_dim, embedding_dim),
            nn.ConvTranspose2d(embedding_dim, 256, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 64
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 128
            nn.ReLU(True),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1), # 256
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, quantized

class Discriminator(nn.Module):
    """PatchGAN Discriminator for sharp edges"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 1, 1), # No stride here
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1) # Output raw logits
        )

    def forward(self, x):
        return self.main(x)