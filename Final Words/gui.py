import cv2
import streamlit as st
from Final_ISL import isl

st.title("Indian Sign Language Recognition")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
if st.button('Start Recognition'):
    while True:
        _, frame = camera.read()
        frame = isl()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
elif st.button('stop'):
    camera.release()
    st.write('Stopped')