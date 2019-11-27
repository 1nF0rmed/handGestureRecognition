from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import pickle
import pandas as pd
import seaborn as sns

# Helper functions
def load_image_files(container_path, dimension=(120, 120)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Indian Sign Language basic dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        print("[LOG] -Open dir: "+str(folders[i]))
        for file in direc.iterdir():
            print("[LOG] --Read file: "+str(file))
            img = imread(file)
            #fd, hog_image = hog(img, orientations=9, pixels_per_cell=(5,5),
            #                        cells_per_block=(1,1), visualize=True, multichannel=False)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
            print("Target: "+str(i))
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def train_model():


    # Load the images processed in prep
    image_dataset = load_image_files("prep_sign_aug_new/")


    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(\
                                        image_dataset.data, \
                                        image_dataset.target, \
                                        test_size=0.3,random_state=109)

    print("[LOG] Loaded dataset.")

    print("Y Test: ")
    print(y_test)
    print("Y train: ")
    print(y_train)

    #exit()

    # SVM training parameters
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # SVM classifier
    svc = svm.SVC(verbose=True)
    clf = GridSearchCV(svc, param_grid)
    #clf = svm.SVC(verbose=True)

    print("[LOG] Setup Classifier.")

    # Traing on data
    clf.fit(X_train, y_train)

    """
    # Load model from Pickle
    clf = pickle.load(open('model_hog_new.save', 'rb'))
    """

    # Pickle model and store
    pickle.dump(clf, open("model_hog_aug.save", 'wb'))

    print("[LOG] Model training complete.")

    print("[LOG] Running on Test set....")

    # Predict on test set
    y_pred = clf.predict(X_test)

    print("[LOG] Complete.")

    # Report the final predictions
    print("Classification report for - \n{}:\n{}\n".format(clf, \
                                metrics.classification_report(y_test, y_pred)))
    print("Final Accuracy: ")
    print(metrics.accuracy_score(y_test, y_pred))

    print(metrics.confusion_matrix(y_test, y_pred))
    y_true = pd.Series(y_test, name='Actual')
    y_hat = pd.Series(y_pred, name='Predicted')

    df_confusion = pd.crosstab(y_true, y_hat)
    plt.figure(figsize=(10,6))
    sns.heatmap(df_confusion, annot=True)
    plt.show()
    #scores = cross_val_score(clf, image_dataset.data, image_dataset.target, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
    train_model()
