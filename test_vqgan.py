import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
from VQ_GAN_model import VQGAN

def load_model(checkpoint_path, device):
    # Initialize model with same parameters as training
    model = VQGAN(num_embeddings=256, embedding_dim=64).to(device)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def process_image(image_path, model, device, output_path):
    # Setup transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            # Get reconstruction and quantization info
            _, x_recon, quantized = model(img_tensor)
            
            # Get codebook indices (optional, for analysis)
            z = model.encoder(img_tensor)
            _, _, encoding_indices = model.quantizer(z)
            
        # Post-process for visualization
        # Denormalize: [-1, 1] -> [0, 1]
        img_tensor = (img_tensor + 1) / 2
        x_recon = (x_recon + 1) / 2
        
        # Concatenate original and reconstruction
        comparison = torch.cat([img_tensor, x_recon], dim=3)
        
        # Save result
        save_image(comparison, output_path)
        print(f"Saved result to {output_path}")
        return encoding_indices
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test VQ-GAN reconstruction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    try:
        model = load_model(args.checkpoint, device)
    except Exception as e:
        print(str(e))
        return

    # Process input
    if os.path.isfile(args.input):
        filename = os.path.basename(args.input)
        output_path = os.path.join(args.output, f"recon_{filename}")
        process_image(args.input, model, device, output_path)
        
    elif os.path.isdir(args.input):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for filename in os.listdir(args.input):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, f"recon_{filename}")
                process_image(input_path, model, device, output_path)
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == "__main__":
    main()
