import cv2
import numpy as np
from keras.models import load_model

# Load the gender classification model
model = load_model("models/model.h5")

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the model
    # resized = cv2.resize(gray, (48, 48))
    resized = cv2.resize(gray, (80, 110))
    # # # Normalize the image
    normalized = resized / 255.0
    # # # Reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (-1, 80, 110, 1))
    return reshaped

# Load the image
image = cv2.imread("images/tony_stark.jpg")

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Predict the gender using the pre-trained model
prediction = model.predict(preprocessed_image)
print(prediction[0][0])
gender = "Male" if prediction[0][0] < 0.15 else "Female"

# Display the gender on the screen
print("Gender:", gender)
