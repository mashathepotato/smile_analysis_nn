import cv2
import dlib

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

img = cv2.imread("images/two_faces.jpg")

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)

for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Draw a rectangle
    print(x1, x2, y1, y2)
    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

    landmarks = predictor(image=gray, box=face)
    print("Landmarks: ", landmarks)

    for n in range(48, 60):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        cv2.circle(img=img, center=(x, y), radius=3, color=(255, 0, 0), thickness=1)

    cv2.circle(img=img, center=(100,100), radius=5, color=(0, 255, 0), thickness=10)

cv2.imshow(winname="Mouth Recognition", mat=img)

cv2.waitKey(delay=0)

cv2.destroyAllWindows()
