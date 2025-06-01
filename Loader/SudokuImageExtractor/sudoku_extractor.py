import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_picture(image):
    """function to greyscale, blur and change the receptive threshold of image"""

    # Translate to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add a gaussian blur. it shall remove noise from the image,
    # but will inflict resolution reduction
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    # blur = cv2.bilateralFilter(gray,9,75,75)

    # Create a threshold image. where a pixel value is either 255 or 0.
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    plt.figure()
    plt.imshow(threshold_img)
    # plt.show()
    return threshold_img

def find_soduko_outline(image, threshold_image):
    """Finding the outline of the sudoku puzzle in the image"""
    image_with_contours = image.copy()

    # Contours can be explained simply as a curve joining all the continuous points (along the boundary),
    # having same color or intensity.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

    plt.figure()
    plt.imshow(image_with_contours)
    #plt.show()
    return contours, image_with_contours

def main_outline(image, contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        # Find the size of the contour.
        area = cv2.contourArea(contour)
        if area >50: # Give some minimal value for the size of the contour
            # Find the perimeter of the contour
            peri = cv2.arcLength(contour, True)
            # Make it simpler, by reducing the size of the perimeter. Shall remove noise in contour description.
            # epsilon (second argument) - maximum distance from contour to approximated contour.
            approx = cv2.approxPolyDP(contour , 0.02* peri, True)
            # If it is a shape of a "rectangle" and its size is the biggest
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    if max_area == 0:
        raise Exception("No 4 point shape of reasonable size found in image.")

    return biggest ,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

if __name__ == '__main__':
    folder = r"../Datasets/Dataset1-SudokuImageDataset-Kaggle/v2_train/v2_train"

    # Choose only a jpg file
    image_name = random.choice(os.listdir(folder))
    while not image_name.endswith(".jpg"):
        image_name = random.choice(os.listdir(folder))

    print("Chosen image file: ", folder + '/' + image_name)

    image = cv2.imread(folder + '/' + image_name)
    image = cv2.resize(image, (450, 450))

    plt.figure()
    plt.imshow(image)
    #plt.show()

    preprocessed_image = preprocess_picture(image)

    contours, contour_1 = find_soduko_outline(image, preprocessed_image)

    black_img = np.zeros((450, 450, 3), np.uint8)
    biggest, maxArea = main_outline(image, contours)
    if biggest.size != 0:
        biggest = reframe(biggest)
        contour_2 = image.copy()
        cv2.drawContours(contour_2, biggest, -1, (0, 255, 0), 10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(image, matrix, (450, 450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(imagewrap)
    #plt.show()
