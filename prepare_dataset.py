import os
import shutil

# Paths
BASE_DIR = "dataset/archive"
TRAIN_COLOR = os.path.join(BASE_DIR, "train_img/train_img/color")
TEST_COLOR = os.path.join(BASE_DIR, "test_img/test_img/color")

REAL_DIR = "dataset/real"
SPOOF_DIR = "dataset/spoof"

# Create output folders
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SPOOF_DIR, exist_ok=True)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".jpg"):
            continue

        src_path = os.path.join(folder_path, filename)

        if "_real" in filename.lower():
            dst_path = os.path.join(REAL_DIR, filename)
            shutil.copy(src_path, dst_path)

        elif "_fake" in filename.lower():
            dst_path = os.path.join(SPOOF_DIR, filename)
            shutil.copy(src_path, dst_path)

# Process train and test folders
process_folder(TRAIN_COLOR)
process_folder(TEST_COLOR)

print("Dataset preparation complete!")
print(f"Real images: {len(os.listdir(REAL_DIR))}")
print(f"Spoof images: {len(os.listdir(SPOOF_DIR))}")
