import streamlit as st
import cv2
from PIL import Image
import anant
import tempfile
import numpy as np
def main_image():
    uploaded_file = st.file_uploader("Choose a file", type = ['jpg','jpeg','jfif','png'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        image = cv2.resize(image, (350,350))
        c1, c2 = st.beta_columns([1,1])
        with c1:
            st.image(image)
        
        kernel_size = st.slider("SET KERNEL SIZE", min_value = 3, max_value = 15, step = 2)
        bgr_image = anant.gaussian_blur(image, kernel_size)
        with c2:
            st.image(bgr_image)


def main_video(type_):
    c1, c2 = st.beta_columns([1,1])
    with c1:
        placehodler_1 = st.empty()
    with c2:
        placehodler_2 = st.empty()
    video = None
    if(type_ == "Upload a video file"):
        uploaded_file = st.file_uploader("Upload file", type = 'mp4')
        if uploaded_file is not None :
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video =  cv2.VideoCapture(tfile.name)
    else:
            video  = cv2.VideoCapture(0)
    
    if video is not None:

        while True:

            ret , frame = video.read()
            if not ret:
                st.error("CHECK YOUR CAMERA, ALLOW THE PROGEAM TO RECORD THE LIVE VIDEO")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placehodler_1.image(frame)
            
            blurredr_frame = anant.gaussian_blur(frame)
            placehodler_2.image(blurredr_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
    