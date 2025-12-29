import os
import json
from pathlib import Path

# Define the path to the cached dataset snapshot
# Note: This hash might change if the dataset is updated, but based on previous ls output:
snapshot_hash = "ef45dfa1d58a83fc375cd025288d4e7f6ff9a5ed"
base_path = Path(os.path.expanduser(f"~/.cache/huggingface/hub/datasets--lil-lab--kilogram/snapshots/{snapshot_hash}"))

def inspect_json_file(file_path, description):
    print(f"\n{'='*20} {description} {'='*20}")
    print(f"Reading: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            # Print first few keys and values
            print("\n--- First 2 entries ---")
            for i, (k, v) in enumerate(data.items()):
                if i >= 2: break
                print(f"Key: {k}")
                print(f"Value: {json.dumps(v, indent=2)}")
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            print("\n--- First 2 items ---")
            for i, item in enumerate(data):
                if i >= 2: break
                print(json.dumps(item, indent=2))
                
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

def list_images(dir_path, description):
    print(f"\n{'='*20} {description} {'='*20}")
    print(f"Listing directory: {dir_path}")
    try:
        files = sorted(list(dir_path.glob("*.png")))
        print(f"Total .png files: {len(files)}")
        print("\n--- First 5 files ---")
        for f in files[:5]:
            print(f.name)
    except Exception as e:
        print(f"Error listing directory: {e}")

if __name__ == "__main__":
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist.")
        print("Please check the snapshot hash in ~/.cache/huggingface/hub/datasets--lil-lab--kilogram/snapshots/")
    else:
        # Inspect Training Texts
        train_whole_json = base_path / "training/texts/train_whole.json"
        inspect_json_file(train_whole_json, "Training Annotations (Whole)")

        train_part_json = base_path / "training/texts/train_part.json"
        inspect_json_file(train_part_json, "Training Annotations (Part)")

        # Inspect Images
        train_black_images = base_path / "training/images/train-black"
        list_images(train_black_images, "Training Images (Black)")
        
        train_color_images = base_path / "training/images/train-color"
        list_images(train_color_images, "Training Images (Color)")
