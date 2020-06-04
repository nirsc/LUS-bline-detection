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

regularized_image = fb_algorithm(gray)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(gray, cmap='gray')
a_line_points = get_horizontal_peaks(regularized_image,num_peaks=1)
b_line_points = get_b_line_peaks(regularized_image,num_peaks=2)
draw_lines(b_line_points, gray)
draw_lines(a_line_points,gray)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(regularized_image,cmap='gray')

plt.show()
