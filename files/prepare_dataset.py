import os
import shutil

BASE_DIR = "dataset/archive"

FOLDERS = [
    ("color", os.path.join(BASE_DIR, "train_img/train_img/color")),
    ("depth", os.path.join(BASE_DIR, "train_img/train_img/depth")),
    ("color", os.path.join(BASE_DIR, "test_img/test_img/color")),
    ("depth", os.path.join(BASE_DIR, "test_img/test_img/depth")),
]

REAL_DIR = "dataset/real"
SPOOF_DIR = "dataset/spoof"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SPOOF_DIR, exist_ok=True)

def process_folder(modality, folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".jpg"):
            continue

        src_path = os.path.join(folder_path, filename)

        new_filename = f"{modality}_{filename}"

        if "_real" in filename.lower():
            shutil.copy(src_path, os.path.join(REAL_DIR, new_filename))

        elif "_fake" in filename.lower():
            shutil.copy(src_path, os.path.join(SPOOF_DIR, new_filename))


for modality, folder in FOLDERS:
    process_folder(modality, folder)

print("Dataset preparation complete (COLOR + DEPTH, no overwrite)")
print(f"Real images: {len(os.listdir(REAL_DIR))}")
print(f"Spoof images: {len(os.listdir(SPOOF_DIR))}")
