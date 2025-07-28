import json
import os
import shutil
from PIL import Image
import yaml
import subprocess
import zipfile

print('Librerie caricate')

# Estrai zip
with zipfile.ZipFile("tennis_court_det_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
    
print('Immagini estratte')

# Percorso originale e nuovo nome
old_name = "data"
new_name = "data_keypoint_detection"

# Rinomina la cartella
if os.path.exists(old_name):
    os.rename(old_name, new_name)
    print(f"Cartella rinominata: {old_name} → {new_name}")
else:
    print(f"Cartella non trovata: {old_name}")

def process_annotations(split: str = "train", move_images: bool = True):
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    json_filename = f"data_{split}.json"
    json_path = os.path.join("data_keypoint_detection", json_filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    folders_to_create = [
        f"data_keypoint_detection/images/{split}",
        f"data_keypoint_detection/labels/{split}"
    ]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    source_image_dir = "data_keypoint_detection/images"
    destination_image_dir = f"data_keypoint_detection/images/{split}"
    destination_label_dir = f"data_keypoint_detection/labels/{split}"

    for annotation in data:
        image_name = annotation['id']
        keypoints = annotation['kps']
        filename = f"{image_name}.png"
        src_path = os.path.join(source_image_dir, filename)
        dst_path = os.path.join(destination_image_dir, filename)

        if move_images:
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
                continue

        image_path = dst_path if move_images else src_path
        try:
            with Image.open(image_path) as img:
                width_pixels, height_pixels = img.size
        except FileNotFoundError:
            print(f"Warning: Unable to open image: {image_path}")
            continue

        # Normalizza e clippa i keypoints
        keypoints_normalized = []
        for x, y in keypoints:
            x_norm = x / width_pixels
            y_norm = y / height_pixels

            clipped = False
            if x_norm < 0:
                x_norm = 0.0
                clipped = True
            elif x_norm > 1:
                x_norm = 1.0
                clipped = True

            if y_norm < 0:
                y_norm = 0.0
                clipped = True
            elif y_norm > 1:
                y_norm = 1.0
                clipped = True

            v = 1 if clipped else 2
            keypoints_normalized.append([x_norm, y_norm, v])

        external_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        subset = [keypoints_normalized[i] for i in external_indices]
        x_coords = [pt[0] for pt in subset]
        y_coords = [pt[1] for pt in subset]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width_box = max_x - min_x
        height_box = max_y - min_y

        label_filename = os.path.join(destination_label_dir, f"{image_name}.txt")
        with open(label_filename, "w") as f:
            bbox_info = [str(v) for v in [0, center_x, center_y, width_box, height_box]]
            flat_keypoints = [str(coord) for pt in keypoints_normalized for coord in pt]
            f.write(" ".join(bbox_info + flat_keypoints) + "\n")

print('Starting preprocess train')
# Per processare i dati di training
process_annotations(split="train", move_images=True)

print('Starting preprocess val')
# Per processare i dati di validazione
process_annotations(split="val", move_images=True)

# move images and labels folders to data_keypoin_detection folder (new folder)
base_path = 'data_keypoint_detection'
destination_root = os.path.join(base_path, "data_keypoint_detection")
os.makedirs(destination_root, exist_ok=True)

src_images = os.path.join(base_path, "images")
dst_images = os.path.join(destination_root, "images")
print(f"Spostando '{src_images}' in '{dst_images}'...")
shutil.move(src_images, dst_images)

src_labels = os.path.join(base_path, "labels")
dst_labels = os.path.join(destination_root, "labels")
print(f"Spostando '{src_labels}' in '{dst_labels}'...")
shutil.move(src_labels, dst_labels)

def write_yolo_yaml(file_path="data_keypoint_detection/data.yaml"):
    data = {
        "path": "data_keypoint_detection",
        "names": ["court"],
        "train": "data_keypoint_detection/images/train",
        "val": "data_keypoint_detection/images/val",
        "kpt_shape": [14,3],
        "flip_idx": [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]
    }

    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"YOLO YAML file scritto in: {file_path}")

# Esempio d'uso
write_yolo_yaml()
print('yaml scritto')

print('Inizio subprocess YOLO11n-pose')
# Avvia comando YOLO pose training
subprocess.run([
    "yolo", "pose", "train",
    "data=data_keypoint_detection/data.yaml",
    "model=yolo11n-pose.yaml",
    "epochs=100",
    "imgsz=1280",
    "patience=10",
    "save=True",
    "save_period=1",
    "project=results_pose",
    "augment=True"
])

print('Terminato!')

