import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocess import mediapipe_detection,landmarks
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,GRU,Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import pyttsx3
def isl_char():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    def draw_styled_landmarks(image,results):
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color = (121,22,76), thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color = (121,44,250), thickness = 2, circle_radius = 2))
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 4),
                                mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2))

    def extract_keypoints(results):
        left_hand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark] ).flatten() if results.left_hand_landmarks else np.zeros(63)
        right_hand = np.array(([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark] )).flatten() if results.right_hand_landmarks else np.zeros(63)
        return np.concatenate([left_hand,right_hand])


    model = keras.models.load_model(r'E:\M.Tech\Dissertation Stage 2 - Final\Experimental\Final Character\Perfect_28_Character_Lrelu.h5')

    tt = {0:'A',1:'B',2:'C',3:'D',4:'E',
        5:'F',6:'G',7:'H',8:'I',9:'J',
        10:'K',11:'L',12:'M',13:'N',14:'O',
        15:'P',16:'Q',17:'R',18:'',19:'S',
        20:' ',21:'T',22:'U',23:'V',24:'W',
        25:'X',26:'Y',27:'Z'}

    def pred(test_path):
        #test_path = np.load(test_path)
        x = model.predict(test_path.reshape(-1,126))
        return tt[np.argmax(x)]
        #op = floor(x[0])

    char = []
    word = []
    frames = 0
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.4) as holistic:
        
        while True: 
            
            _,frame = cap.read()
            image,results = mediapipe_detection(frame,holistic)
            draw_styled_landmarks(image,results)
            

            #cv2.imshow("sign_videos",image)

            keypoints = extract_keypoints(results)
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
            #cv2.putText(image,'Image for{}'.format(pp),(15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
            cv2.imshow("sign_videos",image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

isl_char()