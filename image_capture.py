import cv2
import tkinter as tk
from tkinter import messagebox

time_in_correct_position = 0

def image_capture():
    global time_in_correct_position

    # Function to capture the image and close the interface
    def take_picture_and_close():
        ret, frame = camera.read()
        cv2.imwrite("captured_image.jpg", frame)
        messagebox.showinfo("Success", "Picture captured successfully!")
        root.destroy()

    # Function to process the webcam stream and check for the correct head position and smile
    def process_stream():
        global is_correct_position, time_in_correct_position

        ret, frame = camera.read()

        if not ret:
            messagebox.showerror("Error", "Failed to access the camera.")
            root.quit()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

            if len(smiles) > 0:
                is_correct_position = True
                time_in_correct_position += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 1)
            else:
                is_correct_position = False
                time_in_correct_position = 0
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        if time_in_correct_position >= 20:  # 20 frames at 10ms per frame is approximately 2 seconds
            take_picture_and_close()
        else:
            root.after(10, process_stream)

    # Initialize the camera and cascade classifiers
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    is_correct_position = False
    time_in_correct_position = 0

    # Create the Tkinter root window
    root = tk.Tk()
    root.title("Smile and Pose Detection")

    # Start processing the webcam stream
    process_stream()

    # Run the Tkinter main loop
    root.mainloop()

    # Release the camera and close OpenCV windows when the Tkinter window is closed
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_capture()