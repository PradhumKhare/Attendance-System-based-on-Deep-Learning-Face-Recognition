
import cv2
from PIL import Image
from numpy import asarray

def extract_face(filename, required_size=(160, 160)):
    cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # cascade_eye = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    pixels = cv2.imread(filename)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    face = cascade_face.detectMultiScale(gray, 1.3, 6, minSize=(150, 150))
    for x1,y1,width,height in face :
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    return None

from numpy import expand_dims
from keras.models import load_model



def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # print("--------------------------------0",face_pixels.shape)
    samples = expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    return yhat[0]
import pickle
import time
start = time.time()
model1 = pickle.load(open("SVM_model", 'rb'))
middle = time.time()
# model = pickle.load(open("Facenet_model", 'rb'))
model = load_model('src/facenet_keras.h5')
end = time.time()
print('loading svm model takes  %.3f'%(middle - start),"sec and CVV model takes %.3f"%(end - middle) , "sec .")
from sklearn.preprocessing import LabelEncoder
out_encoder = LabelEncoder()
array =  ['item_1','item_2','item_3','item_4' ]
out_encoder.fit(array)
def predict(Image_Path):
    face_pixels = extract_face(Image_Path,required_size=(160,160))
    if face_pixels is None :
        print("Sorry doesnt recognise Your Face . Try Again")
        return
    face_emb = get_embedding(model, face_pixels)
    samples = expand_dims(face_emb, axis=0)
    yhat_class = model1.predict(samples)
    class_index = yhat_class[0]
    print(out_encoder.inverse_transform(yhat_class)[0])
    print(model1.predict_proba(samples)[0,class_index])
    return out_encoder.inverse_transform(yhat_class)[0],model1.predict_proba(samples)[0,class_index]
predict("src")