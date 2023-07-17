import os
import cv2
import dlib

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Create the output folder if it doesn't exist
output_folder = "dataset/ideal_cropped"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the dataset/ideal folder
image_counter = 1
for filename in os.listdir("dataset/ideal"):
    if filename.endswith(".jpg"):
        image_path = os.path.join("dataset/ideal", filename)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        if len(faces) == 0:
            continue  
        else:
        # Assuming there is only one face in each image
            face = faces[0]
            landmarks = predictor(image=gray, box=face)

        # Check if there are multiple mouths detected
        if len(landmarks.parts()) < 60:
            continue  # Skip images with multiple detected mouths

        # Get mouth landmarks
        x1 = landmarks.part(48).x
        y1 = landmarks.part(51).y
        x2 = landmarks.part(54).x
        y2 = landmarks.part(57).y

        # Crop mouth region
        mouth = img[y1:y2, x1:x2]
        resized_mouth = cv2.resize(mouth, (28, 28))

        # Save cropped mouth image
        output_path = os.path.join(output_folder, f"ideal_cropped{image_counter}.jpg")
        cv2.imwrite(output_path, resized_mouth)

        image_counter += 1

print("Mouth cropping completed")






# import os
# import cv2
# import dlib
# import numpy as np

# # Load the detector and predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# # Create the output folder if it doesn't exist
# output_folder = "dataset/ideal_cropped"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Process each image in the dataset/ideal folder
# image_counter = 1
# for filename in os.listdir("dataset/ideal"):
#     if filename.endswith(".jpg"):
#         image_path = os.path.join("dataset/ideal", filename)
#         img = cv2.imread(image_path)

#         gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#         # Use detector to find landmarks
#         faces = detector(gray)

#         if len(faces) == 0:
#             continue  # Skip images without any detected faces

#         # Assuming there is only one face in each image
#         face = faces[0]
#         landmarks = predictor(image=gray, box=face)

#         # Check if there are multiple mouths detected
#         if len(landmarks.parts()) < 60:
#             continue  # Skip images with multiple detected mouths

#         # Get mouth landmarks
#         x1 = landmarks.part(48).x
#         y1 = landmarks.part(51).y
#         x2 = landmarks.part(54).x
#         y2 = landmarks.part(57).y

#         # If y coords of 48 and 54 (corners of the mouth) aren't the same
#         if landmarks.part(48).y != landmarks.part(54).y:
#             # angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi / 2
#             angle = np.arctan2(landmarks.part(54).y - landmarks.part(48).y, x2 - x1) * 180 / np.pi
#         else:
#             angle = 0
    
#         angle = 0
#         # Rotate the image to align the corners of the mouth
#         # Center of rotation is the left corner
#         # center = (x1 + x2) // 2, (y1 + y2) // 2
#         center = x1, landmarks.part(48).y
#         rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
#         rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

#         # Crop mouth region
#         mouth = rotated_image[y1:y2, x1:x2]
#         resized_mouth = cv2.resize(mouth, (28, 28))

#         # Save cropped mouth image
#         output_path = os.path.join(output_folder, f"ideal_cropped{image_counter}.jpg")
#         cv2.imwrite(output_path, resized_mouth)

#         image_counter += 1

# print("Mouth cropping and rotation completed")
