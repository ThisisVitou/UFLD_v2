import json
import os

# --- CONFIGURATION ---
# Point this to where your TuSimple dataset is extracted
# It should contain the 'clips' folder and the json files.
TUSIMPLE_ROOT = 'D:/code/dataset/tusimple/train_set' 

# The specific JSON files provided in the TuSimple training set
JSON_FILES = [
    'label_data_0313.json',
    'label_data_0531.json',
    'label_data_0601.json'
]

OUTPUT_FILE = 'train_gt.txt'
# ---------------------

def generate_ground_truth():
    print(f"Scanning root directory: {TUSIMPLE_ROOT}")
    total_samples = 0
    missing_files = 0

    with open(OUTPUT_FILE, 'w') as out_f:
        for json_file in JSON_FILES:
            json_path = os.path.join(TUSIMPLE_ROOT, json_file)
            
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found. Skipping.")
                continue
            
            print(f"Processing {json_file}...")
            
            with open(json_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    # The 'raw_file' path in JSON is relative (e.g., "clips/0313-1/...")
                    relative_path = data['raw_file']
                    full_img_path = os.path.join(TUSIMPLE_ROOT, relative_path)
                    
                    # Sanity Check: Does the image actually exist?
                    if os.path.isfile(full_img_path):
                        # We write the relative path and the raw JSON string together
                        # Format: relative_path <space> json_string
                        # Note: We remove the newline from the end of the input line to keep it on one line
                        out_f.write(f"{relative_path} {line.strip()}\n")
                        total_samples += 1
                    else:
                        missing_files += 1

    print("-" * 30)
    print(f"Done! Generated {OUTPUT_FILE}")
    print(f"Total valid samples: {total_samples}")
    if missing_files > 0:
        print(f"Warning: {missing_files} images referenced in JSON were not found on disk.")

if __name__ == '__main__':
    generate_ground_truth()