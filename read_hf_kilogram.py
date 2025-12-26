import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is not installed.")
    print("Please install it by running: pip install datasets")
    sys.exit(1)

def main():
    print("Loading 'lil-lab/kilogram' dataset from Hugging Face...")
    
    try:
        # Load the dataset
        # This will download the dataset if it's not already cached
        dataset = load_dataset("lil-lab/kilogram")
        
        print("\nDataset loaded successfully!")
        print("-" * 50)
        print(f"Dataset structure:\n{dataset}")
        print("-" * 50)
        
        # Get the first split (usually 'train')
        split_names = list(dataset.keys())
        if not split_names:
            print("The dataset is empty.")
            return

        first_split = split_names[0]
        print(f"Accessing the first example from the '{first_split}' split:")
        
        # Get the first example
        example = dataset[first_split][0]
        
        # Print the example nicely
        import json
        # Convert to string if it's not serializable, but usually HF datasets are dicts
        # We use default=str to handle non-serializable objects like images if present
        print(json.dumps(example, indent=2, default=str))
        
    except Exception as e:
        # Check for Windows specific symlink privilege error
        if hasattr(e, 'winerror') and e.winerror == 1314:
            print("\n" + "!" * 60)
            print("ERROR: Windows Permission Error (Symbolic Links)")
            print("!" * 60)
            print("The dataset library tried to create a symbolic link but failed due to insufficient permissions.")
            print("\nSOLUTION:")
            print("1. Enable 'Developer Mode' in Windows Settings (Recommended).")
            print("   (Settings -> Update & Security -> For developers -> Developer Mode)")
            print("2. OR Run this script/terminal as Administrator.")
            print("!" * 60 + "\n")
        
        print(f"An error occurred while loading the dataset: {e}")

if __name__ == "__main__":
    main()
