from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, Response,request
import cv2
import os
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
from tkinter import * 
from PIL import ImageTk,Image


ind = 0

#testing part
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def draw_styled_landmark(image,results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color = (121,22,76), thickness = 2, circle_radius = 4),
                            mp_drawing.DrawingSpec(color = (121,44,250), thickness = 2, circle_radius = 2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 4),
                            mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2))

def extract_keypoint(results):
    left_hand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark] ).flatten() if results.left_hand_landmarks else np.zeros(63)
    right_hand = np.array(([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark] )).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([left_hand,right_hand])

tt = {0:'A',1:'B',2:'C',3:'D',4:'E',
        5:'F',6:'G',7:'H',8:'I',9:'J',
        10:'K',11:'L',12:'M',13:'N',14:'O',
        15:'P',16:'Q',17:'R',18:'',19:'S',
        20:' ',21:'T',22:'U',23:'V',24:'W',
        25:'X',26:'Y',27:'Z'}

model = keras.models.load_model('Perfect_28_Character_relu.h5')

def pred(test_path):
    #test_path = np.load(test_path)
    x = model.predict(test_path.reshape(-1,126))
    return tt[np.argmax(x)]


def gen_frames(paths): 
    char = []
    word = []
    frames = 0
    cap = cv2.VideoCapture(paths)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.4) as holistic:
        while True: 
            _,frame = cap.read()
            if not _:
                break
            else:
                image,results = mediapipe_detection(frame,holistic)
                draw_styled_landmark(image,results)

                keypoints = extract_keypoint(results)
                max_frame = []
                pp = pred(keypoints)
                char.append(pp)
                if len(char)>20:
                    char1 = max(char,key=char.count)
                    if len(char)>20 and char1 == '':
                        word = word[0:-1]
                        char = []
                        continue
                    elif len(char)>60 and char1 == '':
                        word= []
                        char = []
                        continue

                    word.append(char1)
                    char = []
                cv2.rectangle(image,(0,0),(640,40),(0,0,0),-1)
                cv2.putText(image,''.join(word),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def isl(path):
    
    # ax = os.listdir('E:/M.Tech/Final/Word/Mp_Data/')
    actions = ['Afternoon', 'Blind', 'Good', 'Hello', 'Home', 'Marriage', 'Sad', 'Thanks', '_']
    # label_map = {}

    # for a in range(len(ax)):
    #     label_map[ax[a]]=a
    #     actions.append(ax[a])

    # action = np.array(actions)

    model = keras.models.load_model('Perfect_9_GRU.h5')

    x_test = np.zeros((12,30,1662))
    res = model.predict(x_test)

    predictions = [9,9,9,9,9,9,9,9,9,9]

    sequence = []
    sentence = []

    threshold = 0.9
    cap = cv2.VideoCapture(path)
    with mp_holistic.Holistic(min_detection_confidence = 0.6, min_tracking_confidence = 0.6) as holistic:
        while cap.isOpened():
            _,frame = cap.read()
            image,results = mediapipe_detection(frame,holistic)
            draw_styled_landmarks(image,results)


            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis = 0))[0]
                #print(actions[np.argmax(res)])
                #print(res)
                predictions.append(np.argmax(res))
                t = max(res)

            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] >= threshold:
                    #print('....',np.unique(predictions[-10:])[0],'....')
                    if len(sentence) >0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        if actions[np.argmax(res)] == '_':
                            sentence.remove("_")
                            #speak(sentence)
                            sentence = []
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) >5:
                sentence = sentence[-5:]

            cv2.rectangle(image,(0,0),(640,80),(0,0,0),-1)
            cv2.putText(image,' '.join(sentence),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result







app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/character')
def interpret_character():
    return render_template('char.html')

@app.route('/word')
def interpret_words():
    # Add your interpretation logic here
    return render_template('word.html')


@app.route('/live_character')
def live_character():
    path = 0
    return Response(gen_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_character', methods=['POST'])
def video_character():
    file = request.files['video']
    file.save('static/' + file.filename)  # Save the uploaded file to a folder called 'static'
    video_url = 'static/' + file.filename

    return Response(gen_frames(video_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template('display.html', video_url=video_url)



@app.route('/live_word')
def live_word():
    path = 0
    return Response(isl(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_word', methods=['POST'])
def video_word():
    file = request.files['video']
    file.save('static/' + file.filename)  # Save the uploaded file to a folder called 'static'
    video_url = 'static/' + file.filename
    return Response(isl(video_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return render_template('display.html', video_url=video_url)




if __name__ == "__main__":
    app.run()