from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.track(source='rtsp://192.168.5.157/live/0/MAIN', show=True, conf=0.6, classes=[0], persist=True)

# import cv2
#
# # Open the RTSP stream
# cap = cv2.VideoCapture('rtsp://192.168.5.157/live/0/MAIN')
#
# # Check if the stream is opened successfully
# if not cap.isOpened():
#     print("Error: Unable to open RTSP stream.")
#     exit()
#
# # Get the resolution of the RTSP stream
# rtsp_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# rtsp_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("RTSP Stream Resolution:", (rtsp_width, rtsp_height))
#
# # Release the video capture object
# cap.release()
