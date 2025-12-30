import os
import json
import cv2
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
import sys

# Add test directory to path to import find_detail
sys.path.append(os.path.join(os.getcwd(), 'test'))
from find_detail import analyze_tangram_canny, get_shape_and_angle

def process():
    # 1. Setup paths
    output_dir = "preprogress_data/train_black"
    os.makedirs(output_dir, exist_ok=True)
    label_save_path = os.path.join(output_dir, "labels.json")
    
    # Load existing labels if any
    if os.path.exists(label_save_path):
        try:
            with open(label_save_path, 'r') as f:
                all_labels = json.load(f)
                if isinstance(all_labels, list): all_labels = {}
        except:
            all_labels = {}
    else:
        all_labels = {}
    
    # 2. Download JSON
    print("Downloading annotations...")
    try:
        json_path = hf_hub_download(repo_id="lil-lab/kilogram", filename="training/texts/train_whole.json", repo_type="dataset")
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded annotations. Type: {type(annotations)}")
    except Exception as e:
        print(f"Error downloading/loading annotations: {e}")
        annotations = {}
    
    # 3. List Images
    print("Listing images in training/images/train-black/ ...")
    try:
        all_files = list_repo_files(repo_id="lil-lab/kilogram", repo_type="dataset")
        image_files = [f for f in all_files if f.startswith("training/images/train-black/") and f.endswith(".png")]
        print(f"Found {len(image_files)} images.")
    except Exception as e:
        print(f"Error listing files: {e}")
        return

    # 4. Process Loop
    for idx, hf_image_path in enumerate(image_files):
        image_filename = os.path.basename(hf_image_path)
        image_name_no_ext = os.path.splitext(image_filename)[0]
        
        if image_name_no_ext in all_labels:
            print(f"[{idx+1}/{len(image_files)}] Skipping {image_filename} (already processed)")
            continue
            
        print(f"[{idx+1}/{len(image_files)}] Processing {image_filename}...")
        
        try:
            local_image_path = hf_hub_download(repo_id="lil-lab/kilogram", filename=hf_image_path, repo_type="dataset")
        except Exception as e:
            print(f"Error downloading {image_filename}: {e}")
            continue

        # Analyze
        components, original_img = analyze_tangram_canny(local_image_path, visualize=False)
        
        if not components:
            print(f"No components found for {image_filename}")
            continue
            
        # Find annotation
        target_annotation = None
        key_name = image_name_no_ext
        
        if isinstance(annotations, dict):
            if key_name in annotations:
                target_annotation = annotations[key_name]
            elif image_filename in annotations:
                target_annotation = annotations[image_filename]
            elif hf_image_path in annotations:
                target_annotation = annotations[hf_image_path]
        elif isinstance(annotations, list):
            for item in annotations:
                fname = item.get('file_name') or item.get('image_id') or item.get('id')
                if fname and (image_filename in str(fname)):
                    target_annotation = item
                    break
        
        if not target_annotation:
            # print(f"Warning: Annotation for {image_filename} not found.")
            target_annotation = {"text": "Unknown", "original_filename": image_filename}

        # Generate Images
        h, w = original_img.shape[:2]
        
        # Save original
        original_save_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(original_save_path, original_img)
        
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        image_data = {
            "image_path": image_filename, 
            "original_annotation": target_annotation,
            "steps": []
        }
        
        for i, cnt in enumerate(components):
            cv2.drawContours(canvas, [cnt], -1, (0, 0, 0), -1)
            save_name = f"{image_name_no_ext}_step_{i+1}.png"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, canvas)
            
            M = cv2.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            shape_type, rotation = get_shape_and_angle(cnt)
            
            step_entry = {
                "step": i + 1,
                "image_path": save_name,
                "current_object": {
                    "id": i,
                    "type": shape_type,
                    "center": [cx, cy],
                    "rotation": float(f"{rotation:.1f}")
                }
            }
            image_data["steps"].append(step_entry)
            
        all_labels[image_name_no_ext] = image_data
        
        # Save periodically
        if (idx + 1) % 10 == 0:
            with open(label_save_path, 'w') as f:
                json.dump(all_labels, f, indent=2, ensure_ascii=False)
            print(f"Progress saved.")

    # Final save
    with open(label_save_path, 'w') as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    print(f"All done. Labels saved to {label_save_path}")

if __name__ == "__main__":
    process()
