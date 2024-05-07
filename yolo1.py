from multiprocessing import Process
from threading import Thread
from ultralytics import YOLO

def detection_process(sources):
    model = YOLO('yolov8s.pt')
    threads = []
    for source in sources:
        thread = Thread(target=detect_and_show, args=(model, source,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def detect_and_show(model, source):
    results = model.predict(source)
    # Filter for class 0 (humans)
    results = [x for x in results.xyxy[0] if int(x[5]) == 0]
    # Visualize
    for result in results:
        print(result)

if __name__ == '__main__':
    sources = [['rtsp://192.168.5.157/live/0/MAIN'], [0]]  # Add more sources as needed

    processes = []

    for source in sources:
        process = Process(target=detection_process, args=(source,))  # Pass a list containing the source
        process.start()
        processes.append(process)

    for process in processes:
        process.join()