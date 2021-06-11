from numpy.lib.function_base import select
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import anant



def main_image():
    uploaded_file = st.file_uploader("Choose a file", type = ['jpg','jpeg','jfif','png'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.write("ORIGINAL IMAGE")
        image = cv2.resize(image, (350,350))
        c1, c2 = st.beta_columns([1,1])
        with c1:
            st.image(image)
        graycale_image = anant.grayscale(image)
        
        option = {
            "laplacian" : anant.laplacian,
            "scharr" : anant.scharr,
            "perwitt" : anant.perwitt,
            "sobel" : anant.sobel,
            "canny" : anant.canny

        }
        
        st.sidebar.title('Choose filter')
        selection = st.sidebar.radio("Go to", list(option.keys()))
        option = option[selection]
        

        if(selection == "canny"):
            min_, max_ = st.slider(label = "THRESHOLD", min_value = 0, max_value = 255,value = (80,240))
            converted_image = option(graycale_image,min_,max_)
            
        else:
            converted_image = option(graycale_image)

        h , w = converted_image.shape
        edge_detected_image = np.zeros((h,w,3),dtype = np.uint8)
        
        edge_detected_image[:, : , 0] = converted_image
        edge_detected_image[:, : , 1] = converted_image
        edge_detected_image[:, : , 2] = converted_image
        
        with c2:
            st.image(edge_detected_image)


def main_video(type_):
    c1,c2 = st.beta_columns([1,1])
    with c1:
        placehodler_1 = st.empty()
    with c2:
        placehodler_2 = st.empty()

    option = {
            "laplacian" : anant.laplacian,
            "scharr" : anant.scharr,
            "perwitt" : anant.perwitt,
            "sobel" : anant.sobel,
            "canny" : anant.canny

    }
        
    st.sidebar.title('Choose filter')
    selection = st.sidebar.radio("Go to", list(option.keys()))
    option = option[selection]

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
            frame = cv2.resize(frame, (350,350))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placehodler_1.image(frame)
            grayscale_frame = anant.grayscale(frame)
            edge_detected_frame = anant.laplacian(grayscale_frame)
            if selection == "scharr":
                edge_detected_frame = anant.scharr(grayscale_frame)
            elif selection == "perwitt":
                edge_detected_frame = anant.perwitt(grayscale_frame)
            elif selection == "sobel":
                edge_detected_frame = anant.sobel(grayscale_frame)
            elif selection == "canny":
                edge_detected_frame = anant.canny(grayscale_frame)
                
            placehodler_2.image(edge_detected_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()

            
           
        


