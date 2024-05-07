import cv2
import os
import subprocess

video_path = 'H265.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
cap.release()


def main(video_path, output_folder):
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print("Video Information:")
    print("Video Format:", video_path.split(".")[-1])
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration} seconds")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)


# Cut the video
def reduce_video(video_path, output_file, start_time, end_time):

    # FFmpeg command to skip frames and adjust frame rate
    ffmpeg_cmd = f'ffmpeg -i "{video_path}" -ss "{start_time}" -to "{end_time}" -c:v copy -c:a copy "{output_file}"'

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True)
        print(f"Video time length reduced successfully. Saved as {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to reduce video time length: {e}")

    print("Frames extracted and saved.")


def extract_frames(cut_video, output_folder, fps):
    # Open the cut video file
    cap = cv2.VideoCapture(cut_video)
    if not cap.isOpened():
        print("Error: Could not open cut video file.")
        return

    # Read frames and save every 5th and 15th frame of each second
    frame_interval = int(fps)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_interval == 4 or frame_number % frame_interval == 14:
            second = frame_number // frame_interval + 1
            second_folder = os.path.join(output_folder, f"second_{second}")
            os.makedirs(second_folder, exist_ok=True)
            cv2.imwrite(os.path.join(second_folder, f"frame_{frame_number}.jpg"), frame)
        frame_number += 1

    cap.release()


if __name__ == "__main__":
    video_path = "H265.mp4"  # Path to the input video
    output_folder = "../output_folder"  # Output folder to save the modified video and extracted frames
    output_file = "output_file.mp4"
    cut_video = output_file
    main(video_path, output_folder)
    start_time = '00:00:01'
    end_time = '00:00:20'
    reduce_video(video_path, output_file, start_time, end_time)
    extract_frames(cut_video, output_folder, fps)
