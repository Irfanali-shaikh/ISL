import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

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