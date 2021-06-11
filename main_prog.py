#debug
from numpy.lib.function_base import select
import streamlit as st
import grayscale
import bgr
import blur
import Edge_detection
import R_B_G_channel
import thresholding_image
import erosion
import dilation
import Grayscale_inverse

pages = {
    "R-G-B-channels" : R_B_G_channel,
    "Grayscale" : grayscale,
    "Grayscale Inverse" : Grayscale_inverse,
    "BGR scale" : bgr,
    "Blurring" : blur,
    "Thresholding" : thresholding_image,
    "Edge detection" : Edge_detection,
    "Erosion" : erosion,
    "Dilation" : dilation
}

st.sidebar.title('Select')
select = st.sidebar.selectbox("Go To", ["Image" , "Video"])

if(select == "Image"):
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page.main_image()
else:
    type_ = st.sidebar.selectbox("Go to", ["Upload a video file", "Use webcam"])
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page.main_video(type_)
