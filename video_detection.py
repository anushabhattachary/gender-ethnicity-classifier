import cv2
import torch
from ultralytics import YOLO
from deepface import DeepFace

# Load the trained YOLO model
model = YOLO("fairface_yolo_project/gender_ethnicity_model/weights/best.pt")

# Open video file or webcam (0 for webcam)
video_path = "input_video.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO face detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                continue

            # DeepFace analysis
            try:
                analysis = DeepFace.analyze(cropped_face, actions=['gender', 'race'], enforce_detection=False)
                gender = analysis[0]['dominant_gender']
                ethnicity = analysis[0]['dominant_race']
                label = f"{gender}, {ethnicity}"
            except Exception as e:
                label = "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame (optional)
    cv2.imshow("Video Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_path}")
