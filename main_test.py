import cv2
import supervision as sv
from supervision import ColorLookup
from ultralytics import YOLO
import uuid
import os
from datetime import datetime
import threading
from queue import Queue
import queue


class AICore:
    def __init__(self, source_video_path='cctv.mp4', output_dir='cropped_images'):
        self.model = YOLO("yolov8s.pt")
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack

        # Set up annotators
        self.box_annotator = sv.BoundingBoxAnnotator(color_lookup=ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

        # Frame counter
        self.counter = 0

        # Number of frames to skip
        self.skip_frames = 1

        # Source video path
        self.SOURCE_VIDEO_PATH = source_video_path

        # Dictionary to map tracker_id to UUID
        self.tracker_id_to_uuid = {}

        # Define the directory to save the cropped images
        self.output_dir = output_dir

        # Create the directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Set to store unique IDs for which images have been saved
        self.saved_images_ids = set()

        # Initialize running state
        self.running = False
        self.thread = None
        self.frame_queue = Queue()  # Queue to hold frames for display

    def get_uuid_for_tracker_id(self, tracker_id):
        if tracker_id not in self.tracker_id_to_uuid:
            # Generate a UUID and take the first 8 characters
            self.tracker_id_to_uuid[tracker_id] = str(uuid.uuid4()).replace("-", "")
        return self.tracker_id_to_uuid[tracker_id]

    def process_frame(self, results, frame):
        detections = sv.Detections.from_ultralytics(results)

        # Filter only humans
        human_detections = detections[detections.class_id == 0]
        if not human_detections:
            return frame
        human_detections = self.tracker.update_with_detections(human_detections)  # Update tracker with human detections

        # Generate or retrieve UUIDs for each tracker ID
        labels = [f"#{self.get_uuid_for_tracker_id(tracker_id)}" for tracker_id in human_detections.tracker_id]

        # Get the minimum length of the two lists
        min_length = min(len(human_detections.xyxy.tolist()), len(labels))

        # Iterate through the bounding boxes
        for i in range(min_length):  # Only iterate over valid indices
            box = human_detections.xyxy.tolist()[i]
            # Check if the image for this ID has already been saved
            if labels[i] in self.saved_images_ids:
                continue

            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            # Get the current date and time
            now = datetime.now()
            # Format the date and time
            date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            # Save the cropped object as an image in the defined directory
            cv2.imwrite(os.path.join(self.output_dir, labels[i] + '@' + date_time + '.jpg'), crop_object)

            # Add the ID to the set of saved images IDs
            self.saved_images_ids.add(labels[i])

        # Annotate the frame with bounding boxes, labels, and traces
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=human_detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=human_detections, labels=labels)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections=human_detections)
        return annotated_frame

    def display_frames(self):
        while True:
            if not self.running and self.frame_queue.empty():
                break

            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow("Tracking Results", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

    def run(self):
        # Open RTSP stream and start prediction
        results = self.model.predict(source=self.SOURCE_VIDEO_PATH, show=False, stream=True, classes=[0])

        for result in results:
            if not self.running:
                break
            frame = result.orig_img  # Get the original frame
            self.counter += 1
            if self.counter % self.skip_frames != 0:
                continue

            annotated_frame = self.process_frame(result, frame)
            if annotated_frame is None:
                print("No frame returned from process_frame")
                continue

            try:
                self.frame_queue.put_nowait(annotated_frame)
                print("Frame added to queue")
            except queue.Full:
                print("Frame queue is full")
                pass  # Drop the frame if the queue is full

    def start(self):
        if self.running:
            print("AI Core is already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        if not self.running:
            print("AI Core is not running.")
            return
        self.running = False
        if self.thread:
            self.thread.join()
        print("AI Core stopped.")


if __name__ == "__main__":
    ai_core = AICore(source_video_path='cctv.mp4', output_dir='cropped_images')
    ai_core.start()
    ai_core.display_frames()
    cv2.destroyAllWindows()