import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max


def get_intersection_points_of_line_and_image_boundary(peak, image, angles):
    peak_r = peak[0]
    peak_theta = angles[peak[1]]
    center_i, center_j = image.shape[0] // 2, image.shape[1] // 2
    #representing the slope of the line from the origin to the point.
    #switching from polar represatation to (slope,bias) representation.
    slope = np.tan(np.deg2rad(peak_theta - 90))
    point_j = center_j + peak_r * np.cos(np.deg2rad(peak_theta))
    point_i = center_i - peak_r * np.sin(np.deg2rad(peak_theta))
    i_s = np.arange(image.shape[1])
    j_s = point_i + (i_s - point_j) * slope * (-1)  # the y axis is inverted when using i,j image coordinates
    #filtering out the parts of the line that fall out of the picture
    in_image = (j_s > 0) & (j_s < image.shape[0])
    point1 = (j_s[in_image][0], i_s[in_image][0])
    point2 = (j_s[in_image][-1], i_s[in_image][-1])
    
    if np.sum(in_image) <= 1:
        point1 = (0, point_j)
        point2 = (image.shape[0] - 1, point_j)
    
    return point1, point2

"""detects peaks through local maxima
parameters:
min_distance - the number of pixels that need to separate two peaks.
num_peaks - the number of peaks to detect in an image. helps to regulate the number of peaks
detected and prevend false positives
"""
def get_peaks(image, num_peaks, min_distance=5):
    return peak_local_max(image, min_distance=min_distance, num_peaks=num_peaks)

def get_peaks_in_window(image, window_size, angles, num_peaks):
    middle = image.shape[0] // 2
    peaks = get_peaks(image[middle - window_size: middle + window_size, angles], num_peaks)
    for peak in range(num_peaks):
        peaks[peak] = [peaks[peak][0] - window_size, angles[peaks[peak][1]]]
    return peaks

"""Searches for local maxima in a range of angles that correspond to fairly horizontal lines"""
def get_horizontal_peaks(image, num_peaks):
    almost_horizontal = np.arange(75, 105)
    return get_peaks_in_window(image, int(image.shape[0] / 4), angles=almost_horizontal, num_peaks=num_peaks)

"""Searches for local maxima in a range of angles that correspond to fairly vertical lines"""
def get_b_line_peaks(image, num_peaks):
    non_horizontal = np.concatenate((np.arange(0, 75), np.arange(105, 180)))
    return get_peaks_in_window(image, int(image.shape[0] / 16), angles=non_horizontal, num_peaks=num_peaks)


def draw_lines(peaks, image, angles=np.arange(180)):
    for peak in peaks:
        point1, point2 = get_intersection_points_of_line_and_image_boundary(peak, image, angles)
        plt.plot([point1[1], point2[1]], [point1[0], point2[0]])



