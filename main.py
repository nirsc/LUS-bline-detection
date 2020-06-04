from preprocessing import preprocess_image
from reconstruct_radon import fb_algorithm
from line_detection import get_horizontal_peaks, get_b_line_peaks, draw_lines
# import numpy as np
import cv2
import matplotlib.pyplot as plt
vid_file = ''
cap = cv2.VideoCapture('Cov-Atlas-Day+1.avi')
ret, frame = cap.read()
# # Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = preprocess_image(gray)
# plt.imshow(gray,cmap = 'gray')
# plt.show()
regularized_image = fb_algorithm(gray)
plt.imshow(regularized_image,cmap = 'gray')
a_line_points = get_horizontal_peaks(gray, regularized_image)
b_line_points = get_b_line_peaks(gray, regularized_image)
draw_lines(b_line_points)
draw_lines(a_line_points)
plt.show()