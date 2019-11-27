from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from shutil import copyfile
import uuid
import math

PATH_TO_DATA = "sign_lang"
MAX_BLOB_SIZE=20
SAVE_DIR = 'prep_sign_aug_new'

def preprocess(img):
    # kernel for morphological operations
    kernel = np.ones((5,5),np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # Slightly blue image to smooth out noise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Remove the white improper background
    ret,th1 = cv2.threshold(gray.copy(),160,255,cv2.THRESH_BINARY)
    # Perform morphological erode
    th1= cv2.erode(th1,kernel,iterations =2)

    # Use the mask built by th1 and extract only part of gray
    dump = gray.copy()
    dump[th1!=0] = 255

    fd, hog_image = hog(dump, orientations=9, pixels_per_cell=(8,8),block_norm= 'L2',
                            cells_per_block=(2,2), visualize=True, multichannel=False)

    return hog_image

def zoom(img):
    zoom_img = cv2.resize(img, img.shape, fx=1.5, fy=1.5,\
                            interpolation=cv2.INTER_LINEAR)

    return zoom_img

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def shiftImg(img, side="left"):
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]

    if side=="left":
        # Shifting Left
        for i in range(HEIGHT, 1, -1):
            for j in range(WIDTH):
                if (i < HEIGHT-20):
                    img[j][i] = img[j][i-20]
                elif (i < HEIGHT-1):
                    img[j][i] = 0
    else:
        # Shifting Right
        for j in range(WIDTH):
            for i in range(HEIGHT):
                if (i < HEIGHT-20):
                    img[j][i] = img[j][i+20]

    return img

def createPrep():
    # Create a directory to store the preprocessed images
    try:
        os.mkdir(SAVE_DIR)
    except:
        print("[LOG] Prep folder already exists!!!")

    # Loop through each folder and create in prep
    for dir in os.listdir(PATH_TO_DATA):
        # path to prep folder
        dest_dir = os.path.join(SAVE_DIR,dir)

        # Create a directory for the class in prep
        os.mkdir(dest_dir)
        # path to store class images
        temp_dir = os.path.join(PATH_TO_DATA, dir)

        print("[LOG] -Open Dir: "+temp_dir)

        # Loop through each file in path
        for f in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, f)
            dest_path = os.path.join(dest_dir, f)
            #dest_path_flip = os.path.join(dest_dir, str(uuid.uuid4())+f)
            #dest_path_left = os.path.join(dest_dir, str(uuid.uuid4())+f)
            #dest_path_right = os.path.join(dest_dir, str(uuid.uuid4())+f)
            #zoom_path = os.path.join(dest_dir, str(uuid.uuid4())+f)
            #dest_rotate_path = os.path.join(dest_dir, str(uuid.uuid4())+f)

            print("[LOG] ---Reading: "+src_path)

            # Read the file
            img = imread(src_path)
            dump = preprocess(img)

            #flipped
            #flipped_img = np.fliplr(dump.copy())
            #left translation
            #left_img = shiftImg(dump.copy())
            #right translation
            #right_img = shiftImg(dump.copy(), side="right")
            #zoom in
            #zoom_img = zoom(dump.copy())
            #rotate10_img = rotation(dump.copy(), 10)

            # Store the file
            cv2.imwrite(dest_path, dump)
            #cv2.imwrite(dest_path_flip, flipped_img)
            #cv2.imwrite(dest_path_left, left_img)
            #cv2.imwrite(dest_path_right, right_img)
            #cv2.imwrite(zoom_path, zoom_img)
            #cv2.imwrite(dest_rotate_path, rotate10_img)




#img = imread('t-classes/1/DSC_2017.JPG')

#dump = preprocess(img)

#print(img.shape)

""" HOG features extraction
fd, hog_image = hog(dump, orientations=9, pixels_per_cell=(5,5),
                        cells_per_block=(1,1), visualize=True, multichannel=False)

imshow(hog_image)
plt.show()
"""

if __name__ == "__main__":
    createPrep()
