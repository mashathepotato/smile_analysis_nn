# import requests
# from bs4 import BeautifulSoup
# import os
# import cv2
# import dlib
# import numpy as np

# def image_scraper(search_string, im_class, im_folder):
#     # search_string = input("Search: ")

#     search_list = search_string.split()
#     search_string = "+".join(search_list)
#     url = "https://www.google.com/search?q={keywords}&tbm=isch&ved=2ahUKEwi1rbahpYSAAxUumScCHXGRAtQQ2-cCegQIABAA&oq={keywords}&gs_lcp=CgNpbWcQA1AAWPcLYIQNaABwAHgCgAHuAogBvA6SAQcwLjUuMi4ymAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=tQ2sZPX-Ga6ynsEP8aKKoA0".format(keywords=search_string)

#     # url = "https://universe.roboflow.com/marwa-sakhreya/smile_kinde/browse?queryText=&pageSize=548&startingIndex=0&browseQuery=true"

#     htmldata = requests.get(url).text

#     def get_url(htmldata):
#         urls = []
#         soup = BeautifulSoup(htmldata, 'html.parser')
#         count = 0
#         for item in soup.find_all('img'):
#             if count == 0:
#                 count += 1
#                 continue
#             count += 1
#             # print(item['src'])
#             urls.append(item["src"])
#         return urls
        
#     url_list = get_url(htmldata)

#     def download_image(url, save_path):

#         response = requests.get(url)

#         if response.status_code == 200:
#             with open(save_path, "wb") as handle:
#                 handle.write(response.content)
#             print(f"Image downloaded successfully to {save_path}")
#         else:
#             print("Failed to download the image")

#     num = 0
#     for image_url in url_list:
#         # save_path = "raw_data/{im_folder}/{im_class}/{im_class}".format(im_folder=im_folder, im_class=im_class) + str(num) + ".jpg"
#         save_path = "raw_data/raw_test/{im_class}/{im_class}".format(im_class=im_class) + str(num) + ".jpg"
#         download_image(image_url, save_path)
#         num += 1


# def image_cropper(im_folder, im_class, resize=(28, 28)):

#     # raw_data = "raw_data/" + im_folder + im_class
#     # USE THIS WHEN CURATING TEST DATA
#     # raw_data = "raw_data/raw_test/" + im_class
#     raw_data = "raw_data/gum/" + im_class
#     print("Extracting from: ", raw_data)

#     # Load the detector and predictor
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#     # USE THIS FOR TEST DOWNLOAD
#     # output_folder = "dataset/test/" + im_class + "_test"
#     output_folder = "dataset/" + im_folder + im_class + "_cropped"
#     print("Saving to: ", output_folder)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     image_counter = 1
#     for filename in os.listdir(raw_data):
#         flag = False
#         if not (filename.endswith(".jpg") or filename.endswith(".png")):
#             continue

#         image_path = os.path.join(raw_data, filename)
#         img = cv2.imread(image_path)

#         gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#         # Use detector to find landmarks
#         faces = detector(gray)

#         if len(faces) >= 2:
#             flag = True
#             continue

#         if len(faces) == 0:
#             resized_mouth = cv2.resize(img, resize)
        
#         if len(faces) == 1:
#             face = faces[0]
#             landmarks = predictor(image=gray, box=face)

#             x1 = landmarks.part(48).x
#             y1 = landmarks.part(51).y
#             x2 = landmarks.part(54).x
#             y2 = landmarks.part(57).y
#             if landmarks.part(48).y != landmarks.part(54).y:
#                 angle = np.arctan2(landmarks.part(54).y - landmarks.part(48).y, x2 - x1) * 180 / np.pi
#             else:
#                 angle = 0
        
#             # Rotate the image to align the corners of the mouth
#             # Center of rotation is the left corner
#             center = x1, landmarks.part(48).y
#             rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
#             rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

#             # Find new coords
#             # landmarks = predictor(image=gray, box=face)

#             # x1 = landmarks.part(48).x
#             # y1 = landmarks.part(51).y
#             # x2 = landmarks.part(54).x
#             # y2 = landmarks.part(57).y

#             # Crop mouth region
#             adjustment =int(((x2-x1)-(y2-y1))/2)
#             mouth = rotated_image[y1-adjustment:y2+adjustment, x1:x2]
#             resized_mouth = cv2.resize(mouth, resize)
#             # resized_mouth = mouth
            
#         # Skip image with multiple faces
#         if flag:
#             continue

#         # Save cropped mouth image
#         output_path = os.path.join(output_folder, im_class + f"_cropped{image_counter}.jpg")
#         cv2.imwrite(output_path, resized_mouth)

#         image_counter += 1


# search_string = str(input("Search: "))

# if "flat" in search_string or "straight" in search_string:
#     im_folder = "smile arc/"
#     im_class = "flat"
# elif "ideal" in search_string or "perfect" in search_string:
#     im_folder = "smile arc/"
#     im_class = "ideal"
# elif "inverted" in search_string or "reversed" in search_string:
#     im_folder = "smile arc/"
#     im_class = "inverted"
# elif "gummy" in search_string:
#     im_folder = "gum/"
#     im_class = "gummy"
# elif "normal" in search_string or "non" in search_string:
#     im_folder = "gum/"
#     im_class = "normal"
# else:
#     print("That option does not exist")

# # Download raw images to raw_data
# # image_scraper(search_string, im_class, im_folder)


# '''Extracting from im_folder (set to raw_data or raw_data/raw_test)
#     Determines subfolder from search word (gum or smile arc type)
#     Preprocesses all files and downloads to dataset folder

#     BEFORE LAUNCHING:
#     -Determine extraction folder (input images)
#     -Determine cropped download folder (output_folder)
# '''
# image_cropper(im_folder, im_class, resize=(28, 28))



import os
import cv2
import dlib
import numpy as np

def image_cropper(raw_data, output_folder, im_class, resize=(28, 28)):
    print("Extracting from: ", raw_data)

    # Load the detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    print("Saving to: ", output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_counter = 1
    for filename in os.listdir(raw_data):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(raw_data, filename)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        if len(faces) >= 2:
            continue

        if len(faces) == 0:
            resized_mouth = cv2.resize(img, resize)
        
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
        
            # Rotate the image to align the corners of the mouth
            # Center of rotation is the left corner
            center = x1, landmarks.part(48).y
            rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
            rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

            # Crop mouth region
            adjustment = int(((x2-x1)-(y2-y1))/2)
            mouth = rotated_image[y1-adjustment:y2+adjustment, x1:x2]
            resized_mouth = cv2.resize(mouth, resize)
        
        # Save cropped mouth image
        output_path = os.path.join(output_folder, im_class + f"_cropped{image_counter}.jpg")
        cv2.imwrite(output_path, resized_mouth)

        image_counter += 1

# Set the extraction folder and download folder paths here
extraction_folder = "raw_data/gum/ai_normal"
download_folder = "dataset/ai_gum/ai_normal_cropped"
im_class = "ai_normal"

# Call the image_cropper function with the specified paths
image_cropper(extraction_folder, download_folder, im_class, resize=(28, 28))
