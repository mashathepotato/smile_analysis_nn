# # Straight to mouth detection (no face)
# import cv2

# # Load the pre-trained mouth detection cascade
# mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# # Load the image
# # image = cv2.imread("images/minors_welcome.png")
# image = cv2.imread("dataset/flat_cropped/flat_cropped1.jpg")

# # TESTING
# if mouth_cascade.empty():
#     print("EMPTY")
# else:
#     print("NOT EMPTY")

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Perform mouth detection
# mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=34, minSize=(30, 30))

# minNeighbors = 10
# while len(mouths) == 0:
#     # Log value of minNeighbors
#     print("WHILE LOOP WITH minNeighbors = " + str(minNeighbors))
#     mouths = mouth_cascade.detectMultiScale(gray, 
#                                             scaleFactor=1.7, 
#                                             minNeighbors=minNeighbors, 
#                                             minSize=(30, 30))
#     minNeighbors -= 1
#     if minNeighbors == 0:
#         print("No mouth detected")
#         break

# # Draw rectangles around the detected mouths
# for (x, y, w, h) in mouths:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# # Display the image with rectangles
# cv2.imshow('Mouth Recognition', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2

# Load the pre-trained face and mouth detection cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load the image
image = cv2.imread("images/tony_stark.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))

# Iterate over the detected faces
for (x, y, w, h) in faces:

    # TESTING: draw rectangles around faces
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the region of interest (face)
    face_roi = gray[y:y+h, x:x+w]
    
    # Perform mouth detection within the face region
    mouths = mouth_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=11, minSize=(30, 30))
    
    # If no mouth is found
    minNeighbors = 10
    while len(mouths) == 0:
        # Log value of minNeighbors
        print("WHILE LOOP WITH minNeighbors = " + str(minNeighbors))
        mouths = mouth_cascade.detectMultiScale(face_roi, 
                                                scaleFactor=1.7, 
                                                minNeighbors=minNeighbors, 
                                                minSize=(30, 30))
        minNeighbors -= 1
        if minNeighbors == 0:
            print("No mouth detected")
            break

    # Draw rectangles around the mouths
    for (mx, my, mw, mh) in mouths:
        # Adjust the coordinates to the global image space
        mx += x
        my += y
        cv2.rectangle(image, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

# Display the image with rectangles
cv2.imshow('Haar Mouth Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
