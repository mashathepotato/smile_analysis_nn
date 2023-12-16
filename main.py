from image_capture import image_capture
from scrapers.image_cropper import image_cropper
from cnn.sequentialmodel import ImageClassifier


def main():
    # Take a photo of the user
    image_capture()

    # Preprocess the photo to be readable by the model
    image_cropper()

    # Call the preloaded model weights on preprocessed image
    str_path = "C:/Users/realc/OneDrive/Documents/IoM/Code/dataset/gum"
    img_classifier = ImageClassifier(str_path)

    result = img_classifier.analyze_image()

    if result:
        print("Gummy smile detected")
    else:
        print("Normal smile")


if __name__ == "__main__":
    main()
    