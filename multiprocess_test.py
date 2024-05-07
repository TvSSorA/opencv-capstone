import cv2
import os
import subprocess
from multiprocessing import Pool

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return fps, width, height, duration

def process_video(video_path, output_folder):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    print(f"Processing video: {video_path}")
    fps, width, height, duration = get_video_info(video_path)

    print("Video Information:")
    print("Video Format:", video_path.split(".")[-1])
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration} seconds")

    # Create output folder for each video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # Cut the video
    output_file = os.path.join(video_output_folder, f"{video_name}_cut.mp4")
    start_time = '00:00:01'
    end_time = '00:00:40'
    reduce_video(video_path, output_file, start_time, end_time)

    # Extract frames
    extract_frames(output_file, video_output_folder, fps)

def reduce_video(input_file, output_file, start_time, end_time):
    # FFmpeg command to cut the video
    ffmpeg_cmd = f'ffmpeg -i "{input_file}" -ss "{start_time}" -to "{end_time}" -c:v copy -c:a copy "{output_file}"'
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True)
        print(f"Video time length reduced successfully. Saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to reduce video time length: {e}")

def extract_frames(input_file, output_folder, fps):
    # Open the cut video file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open cut video file '{input_file}'.")
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
            frame_name = f"frame_{frame_number}.jpg"
            cv2.imwrite(os.path.join(second_folder, frame_name), frame)
            print(f"Frame {frame_name} extracted and saved.")
        frame_number += 1

    cap.release()


if __name__ == "__main__":
    video_paths = ["H265.mp4", "ani1.mp4", "ani2.mp4"]  # Paths to the input video files
    output_folder = "../output_folder_2"  # Output folder to save the modified videos and extracted frames

    # Create a pool of worker processes
    with Pool(processes=len(video_paths)) as pool:
        # Apply the process_video function to each video path in parallel
        pool.starmap(process_video, [(video_path, output_folder) for video_path in video_paths])
