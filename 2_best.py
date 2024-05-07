import cv2
import os
import subprocess

def calculate_sharpness(image):
    # Convert 2 grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Measure sharpness
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_frames(input_file, output_folder, skip_factor):
    # Open the video
    cap = cv2.VideoCapture(input_file)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Get video information
    print("Video Information:")
    print("Video Format:", input_file.split(".")[-1])
    print(f"FPS: {fps}")
    print(f"Duration: {duration} seconds")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # FFmpeg command to skip frames and adjust frame rate
    output_file = os.path.join(output_folder, "output_file.mp4")
    # Calculate duration and speed
    adjusted_duration = duration * 3/4
    speed_factor = 0.75
    # FFmpeg command to adjust fps, duration and speed
    ffmpeg_cmd = f'ffmpeg -i "{input_file}" -r {skip_factor} -filter:v "setpts={speed_factor}*PTS" -t {adjusted_duration} "{output_file}"'

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True)
        print(f"Video time length reduced successfully. Saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to reduce video time length: {e}")



    # Extract frames from the output video file
    cap = cv2.VideoCapture(output_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the frame index
        current_second = int(frame_number / fps)

        # Create a folder for current second
        second_folder = os.path.join(output_folder, f"second_{current_second}")
        os.makedirs(second_folder, exist_ok=True)

        # Calculate sharpness
        sharpness = calculate_sharpness(frame)

        # Save the frame to the folder with sharpness value
        frame_path = os.path.join(second_folder, f"frame_{frame_number}_sharpness_{sharpness:.2f}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_number += 1

    cap.release()

    # Select two frames with the highest sharpness from each second folder
    for second_folder in os.listdir(output_folder):
        if not os.path.isdir(os.path.join(output_folder, second_folder)):
            continue

        frames = []
        for frame_file in os.listdir(os.path.join(output_folder, second_folder)):
            if frame_file.endswith(".jpg"):
                sharpness = float(frame_file.split("_")[-1][:-4])
                frames.append((frame_file, sharpness))

        # Sort frames by sharpness
        frames.sort(key=lambda x: x[1], reverse=True)

        # Select two frames with highest sharpness
        selected_frames = frames[:2]

        #fMove selected frames to a new folder
        selected_folder = os.path.join(output_folder, f"{second_folder}_selected_frames")
        os.makedirs(selected_folder, exist_ok=True)
        for frame_file, _ in selected_frames:
            src = os.path.join(output_folder, second_folder, frame_file)
            dst = os.path.join(selected_folder, frame_file)
            os.rename(src, dst)

        print(f"Selected two frames with highest sharpness from {second_folder}")

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Input video
    output_folder = "../output_folder"  # Output folder
    skip_factor = 18
    extract_frames(video_path, output_folder, skip_factor)
