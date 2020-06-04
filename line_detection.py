import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import numpy as np
from skimage.feature import peak_local_max
import math
from preprocessing import preprocess_image

cap = cv2.VideoCapture('Cov-Atlas-Day+1.avi')
ret, frame = cap.read()
# # Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
image = preprocess_image(gray)
max_angle = 45
num_peaks = 1


angles = np.arange(180)
pleural_angles = np.arange(75, 105)
horizontal_r = image.shape[0] / 4


def get_intersection_points_of_line_and_image_boundery(bias, slope, image):
    x = np.arange(image.shape[1])
    y = - slope * x + bias
    in_image = (y > 0) & (y < image.shape[0])
    point1 = (y[in_image][0], x[in_image][0])
    # print(point1)
    point2 = (y[in_image][-1], x[in_image][-1])
    return point1, point2


def get_bias_and_slope_by_point_and_slope(peak, image, angles):
    peak_r = peak[0]
    peak_theta = angles[peak[1]]
    slope = np.tan(np.deg2rad(peak_theta - 90))
    center_i, center_j = image.shape[0] // 2, image.shape[1] // 2
    shift_on_center_line = peak_r / np.cos(np.deg2rad(peak_theta))
    point = (center_i, shift_on_center_line + center_j)
    bias = point[0] + point[1] * slope
    return bias, slope


def get_peaks(image, num_peaks, neighborhood_size = 5):
    return peak_local_max(image, min_distance=neighborhood_size, num_peaks=num_peaks)


def get_peaks_in_window(image, window_size, angles, num_peaks):
    middle = image.shape[0] // 2
    peaks = get_peaks(image[middle - window_size: middle + window_size, angles], num_peaks)
    for peak in range(num_peaks):
        peaks[peak] = [peaks[peak][0] - window_size, angles[peaks[peak][1]]]
    return peaks


def get_horizontal_peaks(image, num_peaks):
    almost_horizontal = np.arange(75, 105)
    return get_peaks_in_window(image, int(image.shape[0] / 4), angles=almost_horizontal, num_peaks=num_peaks)


def get_b_line_peaks(image, num_peaks):
    non_horizontal = np.concatenate((np.arange(0, 75), np.arange(105, 180)))
    return get_peaks_in_window(image, int(image.shape[0] / 16), angles=non_horizontal, num_peaks=num_peaks)

def draw_lines(peaks, image, angles = np.arange(180)):
    for peak in peaks:
        bias, slope = get_bias_and_slope_by_point_and_slope(peak, image, angles)
        point1, point2 = get_intersection_points_of_line_and_image_boundery(bias, slope, image)
        plt.plot([point1[1], point2[1]], [point1[0], point2[0]])


sinogram = radon(image, theta=angles, circle=False)
horiontal_lines = get_horizontal_peaks(image=sinogram, num_peaks=1)
b_lines = get_b_line_peaks(image=sinogram, num_peaks=2)



plt.imshow(image, cmap=plt.cm.Greys_r)
draw_lines(horiontal_lines,image)
draw_lines(b_lines,image)




plt.show()

