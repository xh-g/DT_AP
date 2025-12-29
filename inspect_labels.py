import json
import argparse
import os
import sys

def load_labels(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

def list_images(data):
    print(f"Found {len(data)} images in labels file:")
    for key in data.keys():
        # Try to get annotation summary
        anno = data[key].get('original_annotation', [])
        anno_str = str(anno)[:50] + "..." if len(str(anno)) > 50 else str(anno)
        print(f" - {key:<15} | Annotation: {anno_str}")

def show_image_details(data, image_key):
    if image_key not in data:
        print(f"Error: Image '{image_key}' not found.")
        return
    
    info = data[image_key]
    print(f"\n=== Details for {image_key} ===")
    print(f"Image Path: {info.get('image_path')}")
    print(f"Original Annotations:")
    if isinstance(info.get('original_annotation'), list):
        for ann in info.get('original_annotation'):
            print(f"  - {ann}")
    else:
        print(f"  {info.get('original_annotation')}")
        
    steps = info.get('steps', [])
    print(f"\nConstruction Steps ({len(steps)}):")
    print(f"{'Step':<5} {'Type':<15} {'Center':<15} {'Rotation':<10} {'Image Path'}")
    print("-" * 70)
    
    for step in steps:
        s_num = step.get('step')
        img_path = step.get('image_path')
        obj = step.get('current_object', {})
        
        o_type = obj.get('type', 'unknown')
        o_center = str(obj.get('center', []))
        o_rot = str(obj.get('rotation', 0))
        
        print(f"{s_num:<5} {o_type:<15} {o_center:<15} {o_rot:<10} {img_path}")

def search_by_type(data, obj_type):
    print(f"\nSearching for object type: '{obj_type}'")
    found_count = 0
    for img_key, info in data.items():
        for step in info.get('steps', []):
            if step.get('current_object', {}).get('type') == obj_type:
                print(f" - Found in {img_key}, Step {step.get('step')}")
                found_count += 1
    
    if found_count == 0:
        print("No matches found.")
    else:
        print(f"Total occurrences: {found_count}")

def main():
    parser = argparse.ArgumentParser(description="Inspect labels.json content")
    parser.add_argument('--file', default='preprogress_data/train_black/labels.json', help='Path to labels.json')
    parser.add_argument('--list', action='store_true', help='List all image keys')
    parser.add_argument('--image', help='Show details for a specific image key (e.g. page1-82)')
    parser.add_argument('--type', help='Search for images containing a specific object type (e.g. triangle, square)')
    
    # If no args are passed, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        # Also run list by default for convenience
        print("\n--- Default Run (Listing Images) ---")
        args = parser.parse_args(['--list'])
    else:
        args = parser.parse_args()
    
    data = load_labels(args.file)
    if not data:
        return

    if args.list:
        list_images(data)
    elif args.image:
        show_image_details(data, args.image)
    elif args.type:
        search_by_type(data, args.type)

if __name__ == "__main__":
    main()
