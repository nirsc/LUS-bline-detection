from preprocessing import preprocess_image
from reconstruct_radon import fb_algorithm
from line_detection import get_horizontal_peaks, get_b_line_peaks, draw_lines
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

vid_file = ''

def main(args):
    thetas = args.thetas
    num_b_lines = args.num_b_lines
    num_horizontal_lines = args.num_horizontal_lines
    cap = cv2.VideoCapture('Cov-Atlas-Day+1.avi')
    ret, frame = cap.read()
    # # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)
    regularized_image = fb_algorithm(gray, thetas = thetas)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gray, cmap='gray')
    a_line_points = get_horizontal_peaks(regularized_image, num_peaks=num_horizontal_lines)
    b_line_points = get_b_line_peaks(regularized_image, num_peaks=num_b_lines)
    draw_lines(b_line_points, gray)
    draw_lines(a_line_points, gray)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(regularized_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_horizontal_lines', type=int, default=1,
                        help='number of horizontal (pleural and subpleural lines')
    parser.add_argument('--num_b_lines', type=int, default=2,
                        help='number of b-lines to be detected')
    parser.add_argument('--thetas', type = list, default = list(np.arange(180)))
    parser.add_argument('--infile', nargs=1,
                        help="JSON file to be processed",
                        type=argparse.FileType('r'))
    args = parser.parse_args()
    main(args)
