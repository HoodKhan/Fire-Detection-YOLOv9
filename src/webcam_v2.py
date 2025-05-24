from ultralytics import YOLO
import cv2

# Load your trained YOLOv9 model
model = YOLO("runs/detect/train/weights/best.onnx")

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame with confidence threshold
    results = model.predict(frame, conf=0.60, stream=True)  # Set confidence threshold to 80%

    # Plot only detections with confidence > 80%
    for result in results:
        # Get bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Draw bounding boxes and labels for filtered detections
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            label = f"{model.names[int(class_id)]} {confidence:.2f}"  # Create label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Draw label

    # Display the frame with filtered detections
    cv2.imshow("YOLOv9 Webcam Detection (Confidence > 80%)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()