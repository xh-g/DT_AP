import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lpips
from VQ_GAN_model import VQGAN, Discriminator

# --- Hyperparameters ---
LR = 1e-4
BATCH_SIZE = 16 # Adjust based on GPU VRAM
EPOCHS = 50
CODEBOOK_SIZE = 256 # Small codebook for simple geometry
EMBED_DIM = 64
DISC_START = 1000 # Steps before discriminator kicks in
LAMBDA_PERCEPTUAL = 1.0
LAMBDA_GAN = 0.5 # Increased weight for sharpness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
])

# 替换为你的七巧板数据集路径
dataset = datasets.ImageFolder('./tangram_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

vqgan = VQGAN(num_embeddings=CODEBOOK_SIZE, embedding_dim=EMBED_DIM).to(device)
discriminator = Discriminator().to(device)
lpips_loss = lpips.LPIPS(net='vgg').to(device).eval() # Perceptual Loss

opt_vq = torch.optim.Adam(vqgan.parameters(), lr=LR)
opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LR)

# --- Training Loop ---
global_step = 0

print("Start Training...")
for epoch in range(EPOCHS):
    for images, _ in dataloader:
        images = images.to(device)
        
        # 1. Train Generator (Autoencoder)
        vq_loss, x_recon, quantized = vqgan(images)
        
        # Reconstruction Loss (L1 is better for geometric sharpness than L2)
        rec_loss = F.l1_loss(x_recon, images)
        
        # Perceptual Loss
        p_loss = lpips_loss(x_recon, images).mean()
        
        # GAN Loss (Generator part)
        g_loss = 0
        if global_step > DISC_START:
            disc_fake = discriminator(x_recon)
            g_loss = -torch.mean(disc_fake) # Hinge-like or WGAN-like
        
        # Adaptive Weight for GAN (Optional simplified version)
        loss_gen = vq_loss + rec_loss + LAMBDA_PERCEPTUAL * p_loss + LAMBDA_GAN * g_loss
        
        opt_vq.zero_grad()
        loss_gen.backward()
        opt_vq.step()
        
        # 2. Train Discriminator
        loss_disc = torch.tensor(0.0).to(device)
        if global_step > DISC_START:
            disc_real = discriminator(images)
            disc_fake = discriminator(x_recon.detach())
            
            # Hinge Loss
            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            loss_disc = (d_loss_real + d_loss_fake) / 2
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
        if global_step % 100 == 0:
            print(f"Epoch [{epoch}] Step [{global_step}] "
                  f"Loss Gen: {loss_gen.item():.4f} "
                  f"Rec: {rec_loss.item():.4f} "
                  f"Loss Disc: {loss_disc.item():.4f}")
            
        global_step += 1

    # Save checkpoint
    torch.save(vqgan.state_dict(), f"vqgan_epoch_{epoch}.pth")