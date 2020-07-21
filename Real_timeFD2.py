import cv2
# from Face_predictor_2 import predict
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from keras.models import load_model

import pickle
import time
start = time.time()
model1 = pickle.load(open("SVM_model", 'rb'))
middle = time.time()
model = load_model('src/facenet_keras.h5')
end = time.time()
print('loading svm model takes  %.3f'%(middle - start),"sec and CVV model takes %.3f"%(end - middle) , "sec .")
from sklearn.preprocessing import LabelEncoder
out_encoder = LabelEncoder()
array =  ['item_1','item_2','item_3','item_4' ]
out_encoder.fit(array)


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype("float32")
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def predict(face_):
    if face_ is None :
        print("Sorry doesnt recognise Your Face . Try Again")
        return
    face_emb = get_embedding(model, face_)
    samples = expand_dims(face_emb, axis=0)
    yhat_class = model1.predict(samples)
    class_index = yhat_class[0]
    return out_encoder.inverse_transform(yhat_class)[0],model1.predict_proba(samples)[0,class_index]

while True :
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cap.set(3, 480)  # set width of the frame
    cap.set(4, 640)
    ret, frame = cap.read()
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blackwhite = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(blackwhite, scaleFactor=1.3, minNeighbors=6, flags=cv2.CASCADE_SCALE_IMAGE)
    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 4)
        face = frame[y:y+h,x:x+w]
        # face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = asarray(image)
        prediction , probab = predict(face_array)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if probab > 0.9 :
            overlay_text = "%s "% (prediction)
            cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()