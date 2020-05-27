# -*- coding: utf-8 -*-
"""Basic_image_processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YTGBN4GQ55AMRgpp17exnooEeN3LUAlu
"""

def get_margins(arr, th):
  values, counts = np.unique(arr,return_counts= True)
  filtered_values = [values[i] for i in range(len(values)) if counts[i] > th ]
  return np.amin(filtered_values),np.amax(filtered_values)

"""Each frame has borders with ticks and gray background.
This function finds the crops the relevant parts of the image"""
def crop_image(img):
  n_rows = gray.shape[0]
  n_cols = gray.shape[1]
  r_th = int(n_rows/5)
  c_th = int(n_cols/5)
  zero_indices = np.where(gray == 0)
  row_margins = get_margins(zero_indices[0],r_th)
  col_margins = get_margins(zero_indices[1],c_th)
  return gray[row_margins[0]:row_margins[1],col_margins[0]:col_margins[1]]


def center_image(img):
  n_rows = img.shape[0]
  n_cols = img.shape[1]
  padded_img = np.zeros((n_rows*2,n_cols*2))
  row_offset = n_rows
  col_offset = int(n_cols/2)
  padded_img[row_offset:,col_offset:col_offset+n_cols] = img
  return padded_img

# Capture frame-by-frame
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
vid_file = ''
cap = cv2.VideoCapture('vid_file')
ret, frame = cap.read()
# # Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plt.imshow(center_image(crop_image(gray)),cmap = 'gray')