import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class KilogramDataset(Dataset):
    def __init__(self, root_dir, split='train', mode='black', transform=None):
        """
        Args:
            root_dir (str): Path to the snapshot directory (e.g. ~/.cache/huggingface/hub/datasets--lil-lab--kilogram/snapshots/...)
            split (str): 'train' or 'validation' (note: directory structure differs slightly, this is for training part)
            mode (str): 'black' (silhouette) or 'color' (segmented parts)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir).expanduser()
        self.mode = mode
        self.transform = transform
        self.samples = []

        if split == 'train':
            # Path to text annotations
            text_path = self.root_dir / 'training' / 'texts' / 'train_whole.json'
            
            # Path to images
            if mode == 'black':
                img_dir = self.root_dir / 'training' / 'images' / 'train-black'
            elif mode == 'color':
                img_dir = self.root_dir / 'training' / 'images' / 'train-color'
            else:
                raise ValueError("Mode must be 'black' or 'color'")
            
            if not text_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {text_path}")
            
            with open(text_path, 'r') as f:
                data = json.load(f)
                
            # data format: {"tangram_id": ["desc1", "desc2", ...]}
            for tangram_id, descriptions in data.items():
                for idx, desc in enumerate(descriptions):
                    if mode == 'black':
                        # For black images, one image per tangram ID
                        img_name = f"{tangram_id}.png"
                    else:
                        # For color images, one image per description index
                        img_name = f"{tangram_id}_{idx}.png"
                    
                    img_path = img_dir / img_name
                    
                    # Only add if image exists (some might be missing?)
                    if img_path.exists():
                        self.samples.append({
                            'image_path': str(img_path),
                            'text': desc,
                            'id': tangram_id,
                            'idx': idx
                        })
        else:
            # Implement validation/dev split logic if needed
            # The structure for dev/test is different according to README
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        text = sample['text']
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'text': text, 'id': sample['id']}

def visualize_dataset(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        sample = dataset[i]
        ax = axes[i]
        ax.imshow(sample['image'])
        ax.set_title(sample['text'][:20] + "...")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('dataset_preview.png')
    print("Saved dataset_preview.png")

if __name__ == "__main__":
    # Hardcoded path based on your environment
    SNAPSHOT_PATH = "~/.cache/huggingface/hub/datasets--lil-lab--kilogram/snapshots/ef45dfa1d58a83fc375cd025288d4e7f6ff9a5ed"
    
    print(f"Loading dataset from {SNAPSHOT_PATH}")
    
    # Test Black (Silhouette)
    try:
        ds_black = KilogramDataset(SNAPSHOT_PATH, mode='black')
        print(f"Loaded {len(ds_black)} samples (Black/Silhouette)")
        print("First sample:", ds_black[0])
    except Exception as e:
        print(f"Error loading black dataset: {e}")

    # Test Color (Segmented)
    try:
        ds_color = KilogramDataset(SNAPSHOT_PATH, mode='color')
        print(f"Loaded {len(ds_color)} samples (Color/Segmented)")
        print("First sample:", ds_color[0])
    except Exception as e:
        print(f"Error loading color dataset: {e}")
        
    # Visualize
    if 'ds_black' in locals():
        visualize_dataset(ds_black)
