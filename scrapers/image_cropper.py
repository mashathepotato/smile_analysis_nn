import os
import cv2
import dlib
import numpy as np

im_class = "inverted"

folder = "dataset/" + im_class
raw_data = "raw_data/" + im_class

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Create the output folder if it doesn't exist
output_folder = folder + "_cropped"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the dataset/ideal folder
image_counter = 1
for filename in os.listdir(raw_data):
    if filename.endswith(".jpg"):
        image_path = os.path.join(raw_data, filename)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        if len(faces) >= 2:
            continue

        if len(faces) == 0:
            # pass  # Skip images without any detected faces
            continue
        
        if len(faces) == 1:
            face = faces[0]
            landmarks = predictor(image=gray, box=face)

        x1 = landmarks.part(48).x
        y1 = landmarks.part(51).y
        x2 = landmarks.part(54).x
        y2 = landmarks.part(57).y
        if landmarks.part(48).y != landmarks.part(54).y:
            angle = np.arctan2(landmarks.part(54).y - landmarks.part(48).y, x2 - x1) * 180 / np.pi
        else:
            angle = 0
    
        # angle = 0
        # Rotate the image to align the corners of the mouth
        # Center of rotation is the left corner
        # center = (x1 + x2) // 2, (y1 + y2) // 2
        center = x1, landmarks.part(48).y
        rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        # Crop mouth region
        adjustment =int(((x2-x1)-(y2-y1))/2)
        mouth = rotated_image[y1-adjustment:y2+adjustment, x1:x2]
        resized_mouth = cv2.resize(mouth, (28, 28))
        # resized_mouth = mouth

        # Save cropped mouth image
        output_path = os.path.join(output_folder, im_class + f"_cropped{image_counter}.jpg")
        cv2.imwrite(output_path, resized_mouth)

        image_counter += 1

print("Mouth cropping and rotation completed")
