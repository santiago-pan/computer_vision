import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from open_set import open_set


# Find peaks and returns positions
def find_peaks(img, ndivisions, index, show=False):
    height = img.shape[0]

    window = int(height/ndivisions)
    start = window * (index-1)
    end = start + window

    summ = np.sum(img[start:end, :], axis=0)
    peaks_x = argrelextrema(summ, np.greater, axis=0, order=2)[0]
    peaks_y = summ[peaks_x]
    sort_index = np.argsort(peaks_y, axis=0)

    peaks_x = peaks_x[sort_index][::-1]
    peaks_y = peaks_y[sort_index][::-1]

    mid_peak_y = int((start+end)/2)
    points = []
    for p in peaks_x:
        points.append([p, mid_peak_y])

    points = np.float32(points).reshape(-1, 1, 2)

    if (show):
        print('peaks_x: ', peaks_x)
        print('peaks_x shape: ', peaks_x)
        print('peaks_y: ', peaks_y)
        print('peaks_y shape: ', peaks_y.shape)
        print('points: ', points)
        plt.plot(summ)
        plt.plot(peaks_x, peaks_y, 'ro')
        plt.show()

    return points


def process_frame(img, specs):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hls_gray = hls[:, :, 2]
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    edges_filtered = cv2.Canny(gray_filtered, 10, 150)

    # sobelx = cv2.Sobel(hls_gray, cv2.CV_64F, 1, 0)
    # abs_sobelx = np.absolute(sobelx)
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    opening = cv2.morphologyEx(edges_filtered, cv2.MORPH_CLOSE, kernel)

    _, hls_binary = cv2.threshold(edges_filtered, 70, 255, cv2.THRESH_BINARY)

    bird_img = cv2.warpPerspective(hls_binary, M, (bv_width, bv_height))

    final = cv2.hconcat((gray, opening, hls_binary))

    cv2.imshow('gray,opening,hls_binary', final)
    cv2.imshow('bird view', bird_img)

    # Fix maximun values in vertical projection
    ndivisions = 10
    points = [[[0., 0.]]]
    for index in range(0, ndivisions+4):
        cpoints = find_peaks(
            bird_img, ndivisions, index, show=False)

        points = np.concatenate((points, cpoints))

    transformed = cv2.perspectiveTransform(points, M_inverse, dst)

    for point in transformed:
        cv2.circle(img, (int(point[0][0]), int(
            point[0][1])), 2, (255, 0, 0), 3)

    cv2.line(img, tuple(np.array(PA, dtype=int)),
             tuple(np.array(PB, dtype=int)), (0, 0, 255), 3)
    cv2.line(img, tuple(np.array(PB, dtype=int)),
             tuple(np.array(PC, dtype=int)), (0, 0, 255), 3)
    cv2.line(img, tuple(np.array(PC, dtype=int)),
             tuple(np.array(PD, dtype=int)), (0, 0, 255), 3)
    cv2.line(img, tuple(np.array(PD, dtype=int)),
             tuple(np.array(PA, dtype=int)), (0, 0, 255), 3)

    cv2.imshow('image', img)
    cv2.moveWindow('image', 0, 0)


# DEMO FRAMES
path = "assets/Frames/"
# path = "assets/truck_video_001/"
path = "assets/caltech-lanes/cordova1/"
specs = open_set(path)

tilt = specs['tilt']
far_point = specs['far_point']
near_point = specs['near_point']
center_point = specs['near_point']
far_aperture = specs['far_aperture']
near_aperture = specs['near_aperture']

# bird view
bv_width = 800
bv_height = 1200

PA = [center_point-far_aperture+tilt, far_point]
PB = [center_point+far_aperture+tilt, far_point]
PC = [center_point+near_aperture, near_point]
PD = [center_point-near_aperture, near_point]

src = np.array([PA, PB, PC, PD], dtype="float32")

dst = np.array([[0., 0.],  [bv_width, 0.],
                [bv_width, bv_height], [0., bv_height]], dtype="float32")


M = cv2.getPerspectiveTransform(src, dst)
M_inverse = cv2.getPerspectiveTransform(dst, src)

for frame in range(1, specs['frames']):
    img = cv2.imread(path + specs['base'] +
                     str(frame).zfill(specs['zfill']) + specs['ext'])
    process_frame(img, specs)
    cv2.waitKey(specs['fps'])


cv2.waitKey(100)
cv2.destroyAllWindows()
