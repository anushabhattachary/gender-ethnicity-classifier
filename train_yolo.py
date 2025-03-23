from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')  # Use the latest YOLO version available

# Train the model
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='fairface_yolo_project',
    name='gender_ethnicity_model',
    device=0  # Set to 'cpu' if no GPU available
)
