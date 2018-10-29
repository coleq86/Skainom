import cv2
import numpy as np
import os
from skimage import exposure

# function to preprocess the images - reduce image size, equalize histogram and crop largest rectangle which fits the retina
def preprocessing(img_width, img_height, path):
    i = 1
    # for loop to walk through all the files in the folder to which the path points
    for root, dirs, files in os.walk(path):
        for file in files:
            print(str(file))
            # works on extensions ending with g eg. jpeg, jpg etc.
            if file.endswith('g'):
                try:
                    imgpath = os.path.join(root, file)
                    img = cv2.imread(imgpath)
                    # Contrast stretching
                    p2, p98 = np.percentile(img, (2, 98))
                    img = exposure.rescale_intensity(img, in_range=(p2, p98))

                    # Equalization
                    img = exposure.equalize_hist(img)
                    if img.size != 0:
                        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _, th2 = cv2.threshold(grey, 8, 255, cv2.THRESH_BINARY)
                        _, contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        areas = [cv2.contourArea(contour) for contour in contours]
                        max_index = np.argmax(areas)
                        cnt = contours[max_index]
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Ensure bounding rect should be at least 16:9 or taller
                        if w / h > 16 / 9:
                            # increase top and bottom margin
                            newHeight = w / 16 * 9
                            y = y - (newHeight - h) / 2
                            h = newHeight
                        # Crop with the largest rectangle
                        crop = img[int(y):int(y + h), int(x):int(x + w)]
                        resized_img = cv2.resize(crop, (img_width, img_height))
                        print(imgpath)
                        cv2.imwrite(os.path.join(imgpath), resized_img)
                        print("done")
                        print(str(i))
                        i = i + 1
                except:
                    continue


if __name__ == "__main__":
    path = "C:/Users/gaurav_ML/PycharmProjects/Fundus/data/27092018_Unlabeled_Requested_Images/27092018_Requested_Images"
    preprocessing(512, 512, path)