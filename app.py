import streamlit as st
import tensorflow as tf
import os
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/f_model2.h5')
  return model
with st.spinner('Model is being loaded..'):
  new_model=load_model()
st.title("Face Mask Detection")

st.write("""
         Welcome to Face Mask Detection
         """
         )




def getPred(frame):
  
  
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,1.1,4)
  for x,y,w,h in faces:
      roi_gray = gray[y:y+h,x:x+w]
      roi_color = frame[y:y+h,x:x+w]
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      facess = faceCascade.detectMultiScale(roi_gray)
      if len(facess) == 0:
          print("Face not detected")
      else:
          for (ex,ey,ew,eh) in facess:
              face_roi = roi_color[ey:ey+eh,ex: ex+ew]
  final_image = cv2.resize(face_roi,(224,224))
  final_image = np.expand_dims(final_image,axis=0)
  final_image = final_image/255.0
  Predictions = new_model.predict(final_image)

  return Predictions

b = st.selectbox('Select picture',('Pic_1','Pic_2','Pic_3','Pic_4','Pic_5','Pic_6','Pic_7','Pic_8','Pic_9','Pic_10'))
b1 = st.button('Click to Predict :')

if b=='Pic_1':
  frame = cv2.imread("stockMaks.jpg")
  st.image("stockMaks.jpg",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_2':
  frame = cv2.imread("00995.png")
  st.image("00995.png",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_3':
  frame = cv2.imread("f1.jpg")
  st.image("f1.jpg",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_4':
  frame = cv2.imread("f2.jpg")
  st.image("f2.jpg",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_5':
  frame = cv2.imread("00015_Mask.jpg")
  st.image("00015_Mask.jpg",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")

if b=='Pic_6':
  frame = cv2.imread("00089.png")
  st.image("00089.png",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_7':
  frame = cv2.imread("00091_Mask.jpg")
  st.image("00091_Mask.jpg")
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
if b=='Pic_8':
  frame = cv2.imread("00091.png")
  st.image("00091.png",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")

if b=='Pic_9':
  frame = cv2.imread("00021_Mask.jpg")
  st.image("00021_Mask.jpg",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")

if b=='Pic_10':
  frame = cv2.imread("00051.png")
  st.image("00051.png",width=400)
  if b1:
    k = getPred(frame)
    if k > 0.87:
      st.write("No Mask")
    else:
      st.write("Mask")
