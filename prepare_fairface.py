import pandas as pd
import os
import cv2

# Define class mappings
gender_map = {'Male': 0, 'Female': 1}
race_map = {'Asian': 2, 'Black': 3, 'White': 4, 'Hispanic': 5}

# Load CSV
train_csv = pd.read_csv('fairface_dataset/fairface_label_train.csv')
val_csv = pd.read_csv('fairface_dataset/fairface_label_val.csv')

def convert_to_yolo_format(image_path, bbox):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    x_center = (bbox[0] + bbox[2]) / 2 / w
    y_center = (bbox[1] + bbox[3]) / 2 / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return x_center, y_center, width, height

def process_data(df, img_dir, label_dir):
    for index, row in df.iterrows():
        image_name = row['file']
        gender = gender_map[row['gender']]
        race = race_map[row['race']]
        
        # Combine gender and race as single class (optional)
        label = race  # Or any combination logic

        # Assuming FairFace provides bounding boxes; otherwise use face detection
        bbox = [0, 0, 1, 1]  # Replace with actual bounding box

        x_center, y_center, w, h = convert_to_yolo_format(os.path.join(img_dir, image_name), bbox)

        # Copy image
        os.system(f"cp {os.path.join(img_dir, image_name)} fairface_yolo/images/train/{image_name}")

        # Write label
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"{label} {x_center} {y_center} {w} {h}\n")

# Process train and val data
process_data(train_csv, 'fairface_dataset/images', 'fairface_yolo/labels/train')
process_data(val_csv, 'fairface_dataset/images', 'fairface_yolo/labels/val')
