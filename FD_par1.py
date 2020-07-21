from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
# from mtcnn.mtcnn import MTCNN
import cv2
import os
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):

    cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    pixels = cv2.imread(filename)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    face = cascade_face.detectMultiScale(gray, 1.3, 6)

    for x1,y1,width,height in face :
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        if face_array is None :
            continue
        return face_array
    return None


# load images and extract faces for all images in a directory
def load_faces(directory, required_size=(160, 160)):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        image = Image.open(path)
        image = image.resize(required_size)
        face = asarray(image)
        # store
        if face is not None:
            faces.append(face)
        else:
            print("None")
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        # if not os.path.isdir(path):
        #     continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('src')
print(trainX.shape, trainy.shape)
# load test dataset
# testX, testy = load_dataset('E:/img_recogmiser/Test_Set_1/')
# save arrays to one file in compressed format
# savez_compressed('E:/img_recogmiser/faces-dataset.npz', trainX, trainy, testX, testy)
savez_compressed('src/faces-dataset.npz', trainX, trainy)