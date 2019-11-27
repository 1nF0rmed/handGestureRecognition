from flask import Flask, flash, redirect, url_for, request
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
import pickle


UPLOAD_FOLDER = './tmp_files/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None

def gen_random_filename(filename):
    # split extension and name
    b = filename.split(".")

    # Set name to a unique id
    b[0] = str(uuid.uuid4())

    # return the new filename
    return '.'.join(b)

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def getPrediction(file_path):
    # Read the image
    original_img = imread(file_path)

    # Preprocess the image
    hog_image = preprocess(original_img)
    hog_image = resize(hog_image, (120,120), anti_aliasing=True, mode='reflect')

    # Predict hog_image
    pred = model.predict([hog_image.flatten()])

    return pred


@app.route('/api/v1/predictGesture', methods=['POST'])
def handleGesture():

    filename = ""

    # Get the file from the POST Request
    if 'file' not in request.files:
        print 'No file part'
        return redirect(request.url)

    _file = request.files['file']
    if _file and allowed_file(_file.filename):
        filename = gen_random_filename(_file.filename)
        _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # TODO Classify the image
    _class =  getPrediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return "Predicted Class: "+str(_class)


if __name__ == "__main__":

    # Load the model
    global model

    model_filename = "model_hog_aug.save"

    model = pickle.load(open(model_filename, 'rb'))

    # print getPrediction("./test4.jpg")

    # Load the flask api
    app.run(host="0.0.0.0", port=5000)
