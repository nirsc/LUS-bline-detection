import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.feature import peak_local_max
import numpy as np


def get_intersection_points_of_line_and_image_boundary(peak, image, angles):
    peak_r = peak[0]
    peak_theta = angles[peak[1]]
    center_i, center_j = image.shape[0] // 2, image.shape[1] // 2
    slope = np.tan(np.deg2rad(peak_theta - 90))
    point_j = center_j + peak_r * np.cos(np.deg2rad(peak_theta))
    point_i = center_i - peak_r * np.sin(np.deg2rad(peak_theta))
    i_s = np.arange(image.shape[1])
    j_s = point_i + (i_s - point_j) * slope * (-1)  # the y axis is inverted when using i,j image coordinates
    in_image = (j_s > 0) & (j_s < image.shape[0])
    point1 = (j_s[in_image][0], i_s[in_image][0])
    point2 = (j_s[in_image][-1], i_s[in_image][-1])
    if np.sum(in_image) <= 1:
        point1 = (0, point_j)
        point2 = (image.shape[0] - 1, point_j)
    return point1, point2


def get_peaks(image, num_peaks, neighborhood_size=1):
    return peak_local_max(image, min_distance=neighborhood_size, num_peaks=num_peaks, exclude_border=False)


def get_peaks_in_window(image, r_delta, angles, num_peaks):
    middle = image.shape[0] // 2
    peaks = get_peaks(image[middle - r_delta: middle + r_delta, angles], num_peaks)
    for peak in range(num_peaks):
        peaks[peak] = [peaks[peak][0] - r_delta, angles[peaks[peak][1]]]
    return peaks


def get_horizontal_peaks(image, num_peaks):
    almost_horizontal = np.arange(75, 105)
    return get_peaks_in_window(image, int(image.shape[0] / 4), angles=almost_horizontal, num_peaks=num_peaks)


def get_b_line_peaks(image, num_peaks):
    non_horizontal = np.concatenate((np.arange(0, 75), np.arange(105, 180)))
    return get_peaks_in_window(image, int(image.shape[0] / 16), angles=non_horizontal, num_peaks=num_peaks)


def draw_lines(peaks, image, angles=np.arange(180)):
    for peak in peaks:
        point1, point2 = get_intersection_points_of_line_and_image_boundary(peak, image, angles)
        plt.plot([point1[1], point2[1]], [point1[0], point2[0]])


