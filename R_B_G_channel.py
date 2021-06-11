from numpy.core.fromnumeric import resize
from numpy.lib.npyio import BagObj
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
        image = cv2.resize(image,(230,230))
        _,c,_ = st.beta_columns([1,1,1])
        with c:
            st.image(image)
        
        (R , G , B) = anant.diff_channel(image)
        
        col1, col2, col3 = st.beta_columns([1,1,1])

        with col1:
            st.image(R, clamp =True)
        with col2:
            st.image(G,clamp = True)
        with col3:
            st.image(B, clamp = True)


def main_video(type_):
    _,c,_ = st.beta_columns([1,1,1])
    c1,c2,c3 = st.beta_columns([1,1,1])
    with c:
        p = st.empty()
    with c1:
        p1 = st.empty()
    with c2:
        p2 = st.empty() 
    with c3:
        p3 = st.empty()
    
    video = None
    if(type_ == "Upload a video file"):
        uploaded_file = st.file_uploader("Upload file", type = 'mp4')
        if uploaded_file is not None:
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
            frame = cv2.resize(frame, (230,230))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            p.image(frame)
            (R,G,B) = anant.diff_channel(frame)

            p1.image(R)
            p2.image(G)
            p3.image(B)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()

