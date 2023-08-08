import numpy as np
import matplotlib.pyplot as plt
import cv2

def is_whiteish(pixel):
    return pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200

def is_redish(pixel):
    return pixel[0] < 120 and pixel[1] < 60 and pixel[2] > 150

def get_surrounding_pixels(i, j, rows, cols):
    """Return coordinates of surrounding pixels."""
    return [(x, y) for x in range(max(0, i-1), min(rows, i+2))
                   for y in range(max(0, j-1), min(cols, j+2))]

def find_teeth_pixels(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img_rgb.shape
    
    visited = np.zeros((rows, cols), dtype=bool)
    output = np.copy(img_rgb)
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and is_whiteish(img_rgb[i, j]):
                cluster = [(i, j)]
                index = 0
                while index < len(cluster):
                    x, y = cluster[index]
                    for nx, ny in get_surrounding_pixels(x, y, rows, cols):
                        if not visited[nx, ny] and (is_whiteish(img_rgb[nx, ny]) or is_redish(img_rgb[nx, ny])):
                            cluster.append((nx, ny))
                            visited[nx, ny] = True
                    index += 1

                # If the cluster size is large enough, draw the border
                if len(cluster) > 0:  # 50 is an arbitrary threshold and might need adjustments
                    for x, y in cluster:
                        for nx, ny in get_surrounding_pixels(x, y, rows, cols):
                            if is_redish(img_rgb[nx, ny]):
                                output[x, y] = [0, 0, 255]

    plt.imshow(output)
    plt.show()

find_teeth_pixels('dataset/smile arc/ideal_cropped/ideal_cropped3.jpg')

# import numpy as np
# from PIL import Image

# def is_whitish(pixel):
#     pass

# def find_teeth(img_path):
#     img = Image.open(img_path)
#     arr = np.array(img)
#     print(arr)

# find_teeth('dataset/smile arc/ideal_cropped/ideal_cropped3.jpg')