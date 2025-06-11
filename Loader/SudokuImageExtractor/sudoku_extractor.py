import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SudokuImageExtractor.digit_recognition.digit_classification_model_training_and_using import predict_digits

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

def find_contours(image, threshold_image):
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

    return contours

def find_soduko_outline(image, contours):
    biggest_contour = np.array([])
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
                biggest_contour = approx
                max_area = area

    if max_area == 0:
        plt.figure()
        plt.imshow(image)
        plt.show()
        raise Exception("No 4 point shape of reasonable size found in image.")

    image_with_biggest_simplified_contour = image.copy()
    cv2.drawContours(image_with_biggest_simplified_contour, [biggest_contour], -1, (0, 255, 0), 3)

    plt.figure()
    plt.imshow(image_with_biggest_simplified_contour)
    # plt.show()

    return biggest_contour

def get_edges_coords_from_soduko_contour(image, soduko_outline_contour):
    soduko_outline_contour = soduko_outline_contour.reshape((4, 2))
    soduko_rectangle_edges_coords = np.zeros((4,1,2),dtype = np.int32)
    add = soduko_outline_contour.sum(1)
    soduko_rectangle_edges_coords[0] = soduko_outline_contour[np.argmin(add)]
    soduko_rectangle_edges_coords[3] = soduko_outline_contour[np.argmax(add)]
    diff = np.diff(soduko_outline_contour, axis =1)
    soduko_rectangle_edges_coords[1] = soduko_outline_contour[np.argmin(diff)]
    soduko_rectangle_edges_coords[2] = soduko_outline_contour[np.argmax(diff)]

    image_with_soduko_edges = image.copy()
    cv2.drawContours(image_with_soduko_edges, soduko_rectangle_edges_coords, -1, (0, 255, 0), 10)
    plt.figure()
    plt.imshow(image_with_soduko_edges)
    # plt.show()

    return soduko_rectangle_edges_coords

def get_extracted_soduko_image(image, soduko_contour):
    pts1 = np.float32(soduko_contour)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imagewrap = cv2.warpPerspective(image, matrix, (450, 450))
    imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(imagewrap)
    # plt.show()

    return imagewrap

def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)

    # plt.figure()
    # plt.imshow(boxes[58])
    # plt.show()

    return boxes

def crop_cell(cells):
    """The sudoku_cell's output includes the boundaries this could lead to misclassifications by the model"""
    cells_cropped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        cells_cropped.append(img)

    #plt.figure()
    #plt.imshow(cells_cropped[58])
    #plt.show()

    return cells_cropped

def get_image_wrap(image):
    image = cv2.resize(image, (450, 450))

    preprocessed_image = preprocess_picture(image)

    contours = find_contours(image, preprocessed_image)

    soduko_outline = find_soduko_outline(image, contours)

    soduko_rectangle_edges_coords = get_edges_coords_from_soduko_contour(image, soduko_outline)

    imagewrap = get_extracted_soduko_image(image, soduko_rectangle_edges_coords)

    return imagewrap

def extract_soduko_from_image(image):
    imagewrap = get_image_wrap(image)

    plt.figure()
    plt.imshow(imagewrap)
    #plt.show()

    sudoku_cells = splitcells(imagewrap)

    sudoku_cell_cropped = crop_cell(sudoku_cells)

    soduku_prediction = predict_digits(sudoku_cell_cropped)

    # print(soduku_prediction)

    return soduku_prediction


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    folder = os.path.join(base_dir, "../Datasets/Dataset1-SudokuImageDataset-Kaggle/v2_train/v2_train")

    # Choose only a jpg file
    #image_name = random.choice(os.listdir(folder))
    #while not image_name.endswith(".jpg"):
    #    image_name = random.choice(os.listdir(folder))
    image_name = "image1055.jpg"

    print("Chosen image file: ", folder + '/' + image_name)

    image = cv2.imread(folder + '/' + image_name)

    plt.figure()
    plt.imshow(image)
    plt.show()

    board = extract_soduko_from_image(image)

    print(board)






