import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import lpips
import wandb
from VQ_GAN_model import VQGAN, Discriminator

# --- Hyperparameters ---
LR = 5e-5  # Decreased from 1e-4 for finer details
BATCH_SIZE = 48  # 充分利用 16GB 显存
EPOCHS = 150 # Increased from 50 to 150
CODEBOOK_SIZE = 256 
EMBED_DIM = 64
DISC_START = 1000 
LAMBDA_PERCEPTUAL = 1.0
LAMBDA_GAN = 0.8 # Reverted to 0.8 for stability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
# 针对七巧板几何特性的数据增强 (降低难度)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5), # 水平翻转
    transforms.RandomVerticalFlip(p=0.5),   # 垂直翻转
    # 降低旋转幅度到 10 度，避免模型难以收敛
    transforms.RandomRotation(degrees=10, fill=255), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
])

# Load all images
dataset = datasets.ImageFolder('./preprogress_data', transform=transform)
print(f"Training on {len(dataset)} images from ./preprogress_data")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

vqgan = VQGAN(num_embeddings=CODEBOOK_SIZE, embedding_dim=EMBED_DIM).to(device)
discriminator = Discriminator().to(device)
lpips_loss = lpips.LPIPS(net='vgg').to(device).eval() 

opt_vq = torch.optim.Adam(vqgan.parameters(), lr=LR)
opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LR)

# 修改 1: 使用新的 GradScaler API
scaler = torch.amp.GradScaler('cuda')

# --- WandB Setup ---
wandb.init(project="vq_gan_train", config={
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "codebook_size": CODEBOOK_SIZE,
    "embed_dim": EMBED_DIM,
    "disc_start": DISC_START,
    "lambda_perceptual": LAMBDA_PERCEPTUAL,
    "lambda_gan": LAMBDA_GAN
})

# --- Training Loop ---
global_step = 0

print("Start Training...")
for epoch in range(EPOCHS):
    for images, _ in dataloader:
        images = images.to(device)
        
        # ===========================
        # 1. Train Generator (VQGAN)
        # ===========================
        opt_vq.zero_grad()
        
        # 修改 2: 使用新的 autocast API
        with torch.amp.autocast('cuda'):
            vq_loss, x_recon, quantized = vqgan(images)
            
            # Reconstruction Loss
            rec_loss = F.l1_loss(x_recon, images)
            
            # Perceptual Loss
            p_loss = lpips_loss(x_recon, images).mean()
            
            # GAN Loss (Generator part)
            g_loss = 0
            if global_step > DISC_START:
                disc_fake = discriminator(x_recon)
                g_loss = -torch.mean(disc_fake) 
            
            loss_gen = vq_loss + rec_loss + LAMBDA_PERCEPTUAL * p_loss + LAMBDA_GAN * g_loss
        
        # 修改 3: 使用 scaler 进行反向传播和优化
        scaler.scale(loss_gen).backward()
        scaler.step(opt_vq)
        scaler.update()
        
        # ===============================
        # 2. Train Discriminator
        # ===============================
        loss_disc = torch.tensor(0.0).to(device)
        if global_step > DISC_START:
            opt_disc.zero_grad()
            
            with torch.amp.autocast('cuda'):
                disc_real = discriminator(images)
                # 注意: x_recon 需要 detach，否则梯度会传回生成器
                disc_fake = discriminator(x_recon.detach())
                
                d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                loss_disc = (d_loss_real + d_loss_fake) / 2
            
            scaler.scale(loss_disc).backward()
            scaler.step(opt_disc)
            scaler.update()
            
        # --- Logging ---
        if global_step % 100 == 0:
            print(f"Epoch [{epoch}] Step [{global_step}] "
                  f"Loss Gen: {loss_gen.item():.4f} "
                  f"Rec: {rec_loss.item():.4f} "
                  f"Loss Disc: {loss_disc.item():.4f}")
            
            wandb.log({
                "epoch": epoch,
                "global_step": global_step,
                "loss_gen": loss_gen.item(),
                "rec_loss": rec_loss.item(),
                "p_loss": p_loss.item(),
                "g_loss": g_loss if isinstance(g_loss, int) else g_loss.item(),
                "loss_disc": loss_disc.item(),
                "vq_loss": vq_loss.item()
            })
            
        if global_step % 500 == 0:
            with torch.no_grad():
                orig = (images[:4] + 1) / 2
                recon = (x_recon[:4] + 1) / 2
                comparison = torch.cat([orig, recon], dim=0)
                grid = make_grid(comparison, nrow=4)
                wandb.log({"reconstruction": wandb.Image(grid, caption="Top: Original, Bottom: Recon")})
            
        global_step += 1

    # Save checkpoint
    torch.save(vqgan.state_dict(), f"vqgan_epoch_{epoch}.pth")

wandb.finish()