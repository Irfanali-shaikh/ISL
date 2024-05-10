import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from preprocess import mediapipe_detection,landmarks,draw_styled_landmarks,extract_keypoints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import pyttsx3

def speak(word):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voices',voices[0].id)
    engine.setProperty('rate',150)
    engine.say(word)
    engine.runAndWait()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def isl():
    ax = os.listdir('E:/M.Tech/Final/Word/Mp_Data/')
    actions = []
    label_map = {}

    for a in range(len(ax)):
        label_map[ax[a]]=a
        actions.append(ax[a])

    action = np.array(actions)

    model = keras.models.load_model('Perfect_9_GRU.h5')

    x_test = np.zeros((12,30,1662))
    res = model.predict(x_test)

    predictions = [9,9,9,9,9,9,9,9,9,9]

    sequence = []
    sentence = []

    threshold = 0.9

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence = 0.6, min_tracking_confidence = 0.6) as holistic:
        while cap.isOpened():
            _,frame = cap.read()
            image,results = mediapipe_detection(frame,holistic)
            draw_styled_landmarks(image,results)

            print(results)


            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis = 0))[0]
                print(actions[np.argmax(res)])
                print(res)
                predictions.append(np.argmax(res))
                t = max(res)

            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    print('....',np.unique(predictions[-10:])[0],'....')
                    if len(sentence) >0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        if actions[np.argmax(res)] == '_':
                            sentence.remove("_")
                            speak(sentence)
                            sentence = []
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) >5:
                sentence = sentence[-5:]

            cv2.rectangle(image,(0,0),(640,40),(0,0,0),-1)
            cv2.putText(image,' '.join(sentence),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            return image
            #cv2.imshow("sign_videos",image)
            #cv2.imshow("sign_videos 1",hsv)

            #if cv2.waitKey(10) & 0xFF == ord('q'):
                #break


    #cap.release()
    #cv2.destroyAllWindows()


isl()

