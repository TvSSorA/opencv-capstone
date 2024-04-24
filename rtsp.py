import cv2
from ultralytics import YOLO
import torch


# Load the YOLOv8 model
model = YOLO ('yolov8s.pt')

cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If frame capture is successful
    if ret:
        # Convert the frame to RGB format (YOLOv8 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PyTorch tensor
        results = model(frame_rgb)

        # Process detections (if any)
        if len(results) > 0:  # Check if detections are present (using len())
            # Get the first detection (you can loop through all detections if needed)
            detection = results[0]  # Access the first element (detection dictionary)

            # Extract object information (class, confidence score, bounding box coordinates)
            try:
                # Attempt to access data using dictionary keys (more robust)
                x_min, y_min, x_max, y_max, conf, class_id, name = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax'], detection['conf'], detection['class'], detection['name']
            except KeyError:  # Handle potential missing keys
                print("Warning: Missing detection data in results")
                continue

            # Convert bounding box coordinates to integers (required for OpenCV drawing)
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # Draw a bounding box around the detected object
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Display the object class and confidence score above the bounding box
            text = f"{name} ({conf:.2f})"  # Format: "class (confidence score)"
            cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow('YOLOv8 Object Detection', frame)

        # Handle the warning (if applicable):
        print("Warning:", "AVCaptureDeviceTypeExternal is deprecated")

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        print("Error: Frame capture failed")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()