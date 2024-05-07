import cv2
import os


# Function to calculate the blur score of a frame
def calculate_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def select_best_frames(video_path, output_folder):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the video's frame rate and total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables
    best_frames = {}
    current_second = 0
    prev_second = -1

    # Output folder to save best frames
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the video frames
    while True:
        # Capture the current frame
        ret, frame = cap.read()

        # Break if the frame couldn't be captured
        if not ret:
            break

        # Calculate the current second
        current_second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)

        # If we entered a new second
        if current_second != prev_second:
            # Calculate the blur score for the current frame
            blur_score = calculate_blur_score(frame)

            # If this is the first frame of the second or the blur score is lower than the previous best frame
            if current_second not in best_frames or blur_score < calculate_blur_score(best_frames[current_second]):
                best_frames[current_second] = frame.copy()

            prev_second = current_second

    # Release the video capture
    cap.release()

    # Save the best frames to separate folders
    for second, frame in best_frames.items():
        folder_name = os.path.join(output_folder, f"second_{second}")
        os.makedirs(folder_name, exist_ok=True)
        frame_path = os.path.join(folder_name, f"best_frame.jpg")
        cv2.imwrite(frame_path, frame)

    print("Best frames saved successfully.")


# Example usage
video_path = "H265.mp4"  # Path to the input video
output_folder = "../best_frames"  # Output folder to save the best frames
select_best_frames(video_path, output_folder)
