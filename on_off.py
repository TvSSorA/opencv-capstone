from main_test import AICore
import cv2
import threading

ai_core = AICore(source_video_path='cctv.mp4', output_dir='cropped_images')


def display_frames():
    while True:
        if not ai_core.running and ai_core.frame_queue.empty():
            break

        if not ai_core.frame_queue.empty():
            frame = ai_core.frame_queue.get()
            yield frame


def user_interface():
    while True:
        user_input = input("Enter 'on' to start the AI core, 'off' to stop it, or 'quit' to exit: ").strip().lower()
        if user_input == 'on':
            ai_core.start()
        elif user_input == 'off':
            ai_core.stop()
        elif user_input == 'quit':
            print("Exiting...")
            ai_core.stop()
            break
        else:
            print("Invalid input. Please enter 'on', 'off', or 'quit'.")


if __name__ == "__main__":
    display_generator = display_frames()

    user_interface_thread = threading.Thread(target=user_interface)
    user_interface_thread.start()

    frame = None
    while True:
        try:
            frame = next(display_generator, frame)
            if frame is not None:
                cv2.imshow("Tracking Results", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ai_core.stop()
                break
        except StopIteration:
            break

    cv2.destroyAllWindows()

    user_interface_thread.join()