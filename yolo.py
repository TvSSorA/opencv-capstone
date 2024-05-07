from multiprocessing import Process
from ultralytics import YOLO

def detection_process(sources):
    model = YOLO('yolov8s.pt')
    for source in sources:
        model.predict(source=source, show=True, conf=0.6, classes=[0])

if __name__ == '__main__':
    sources = ['rtsp://192.168.5.157/live/0/MAIN', 0]  # Add more sources as needed

    # Define the number of processes you want to run
    num_processes = len(sources)

    processes = []

    for source in sources:
        process = Process(target=detection_process, args=([source],))  # Pass a list containing the source
        process.start()
        processes.append(process)

    for process in processes:
        process.join()