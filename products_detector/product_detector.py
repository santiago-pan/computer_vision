from __future__ import print_function
import numpy as np
import cv2
import json
import math


def init_feature():
    detector = cv2.AKAZE_create()
    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(keyPointsImageObject, keyPointsImageScene, matches, ratio=0.75):
    NO_FILTER_FLAG = False
    validKeyPointsImageObject, validKeyPointsImageScene, validMatches = [], [], []
    for match in matches:
        # Compare first and second match. If distance1 < distance2*0.75
        # And return only those that met the formula.
        if NO_FILTER_FLAG or len(match) == 2 and match[0].distance < match[1].distance * ratio:
            validMatch = match[0]
            validKeyPointsImageObject.append(
                keyPointsImageObject[validMatch.queryIdx])
            validKeyPointsImageScene.append(
                keyPointsImageScene[validMatch.trainIdx])
            validMatches.append(validMatch)
    # Get the points of the filtered matches
    validMatchPointsImageObject = np.float32(
        [kp.pt for kp in validKeyPointsImageObject])
    # Get the points of the filtered matches
    validMatchPointsImageScene = np.float32(
        [kp.pt for kp in validKeyPointsImageScene])
    kpPairs = zip(validKeyPointsImageObject, validKeyPointsImageScene)
    return validMatchPointsImageObject, validMatchPointsImageScene, list(kpPairs), validMatches


def explore_match(
        colorMap,
        detectionBoxes,
        imageObject,
        imageScene,
        kpPairs,
        heightOffsetOnScene,
        widthOffsetOnScene,
        transformationStatus=None,
        transformationMatrixH=None):
    heightImageObject, widthImageObject = imageObject.shape[:2]
    heightImageScene, widthImageScene = imageScene.shape[:2]
    vis = np.zeros((max(heightImageObject, heightImageScene),
                    widthImageObject+widthImageScene), np.uint8)
    vis[:heightImageObject, :widthImageObject] = imageObject
    vis[:heightImageScene, widthImageObject:widthImageObject +
        widthImageScene] = imageScene
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if transformationMatrixH is not None:
        corners = np.float32([[0, 0], [widthImageObject, 0], [
                             widthImageObject, heightImageObject], [0, heightImageObject]])
        cornersColorMap = np.float32([[0, 0], [widthImageObject, 0], [
                                     widthImageObject, heightImageObject], [0, heightImageObject]])
        corners = np.int32(cv2.perspectiveTransform(
            corners.reshape(1, -1, 2),
            transformationMatrixH).reshape(-1, 2) + (widthImageObject + widthOffsetOnScene, heightOffsetOnScene))
        cornersColorMap = np.int32(cv2.perspectiveTransform(cornersColorMap.reshape(
            1, -1, 2), transformationMatrixH).reshape(-1, 2) + (0 + widthOffsetOnScene, 0 + heightOffsetOnScene))
        cv2.polylines(vis, [corners], True, (255, 0, 0), 4)

        detectionBoxes.append(cornersColorMap)

    if transformationStatus is None:
        transformationStatus = np.ones(len(kpPairs), np.bool_)
    p1, p2 = [], []
    pmap1, pmap2 = [], []

    for kpp in kpPairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(
            kpp[1].pt) + [widthImageObject + widthOffsetOnScene, 0 + heightOffsetOnScene]))

        pmap1.append(np.int32(kpp[0].pt))
        pmap2.append(
            np.int32(np.array(kpp[1].pt) + [widthOffsetOnScene, heightOffsetOnScene]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)

    COLOR = 0
    for (x1, y1), (x2, y2), inlier in zip(pmap1, pmap2, transformationStatus):
        if inlier:
            colorMap[y2, x2] = COLOR

            colorMap[y2-1, x2] = COLOR
            colorMap[y2+1, x2] = COLOR

            colorMap[y2-1, x2-1] = COLOR
            colorMap[y2+1, x2-1] = COLOR

            colorMap[y2-1, x2+1] = COLOR
            colorMap[y2+1, x2+1] = COLOR

            colorMap[y2+1, x2-1] = COLOR
            colorMap[y2-1, x2-1] = COLOR

            colorMap[y2+1, x2+1] = COLOR
            colorMap[y2-1, x2+1] = COLOR

            colorMap[y2+1, x2+1] = COLOR
            colorMap[y2+1, x2-1] = COLOR

            colorMap[y2-1, x2+1] = COLOR
            colorMap[y2-1, x2-1] = COLOR

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, transformationStatus):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, transformationStatus):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    # Display detection result
    rfactor = 0.2
    cv2.imshow('scene', cv2.resize(vis, None, fx=rfactor,
                                   fy=rfactor, interpolation=cv2.INTER_CUBIC))

    return vis


def match_and_draw(
        colorMap,
        detectionBoxes,
        keyPointsImageObject,
        keyPointsImageScene,
        descriptorsImageObject,
        descriptorsImageScene,
        heightOffsetOnScene,
        widthOffsetOnScene):

    # rawMatches -> Array of DMatch (queryIdx, trainIdx, imgIdx, distance)
    # k = 2, means that we get two matches for every two descriptors
    rawMatches = matcher.knnMatch(
        descriptorsImageObject, trainDescriptors=descriptorsImageScene, k=2)

    # Filter valid matches
    validMatchPointsImageObject, validMatchPointsImageScene, kpPairs, validMatches = filter_matches(
        keyPointsImageObject, keyPointsImageScene, rawMatches)

    # If there is at least MIN_MATCHES matched points
    MIN_MATCHES = 24  # How magic is this?
    if len(validMatchPointsImageObject) >= MIN_MATCHES:

        # Find a poligon that represents the keypoints
        transformationMatrixH, transformationStatus = cv2.findHomography(
            validMatchPointsImageObject, validMatchPointsImageScene, cv2.RANSAC, 5.0)

        if (np.sum(transformationStatus) > MIN_MATCHES):
            print('Inliers: %d' % (np.sum(transformationStatus)))
            vis = explore_match(colorMap, detectionBoxes, imageObject, imageScene, kpPairs, heightOffsetOnScene,
                                widthOffsetOnScene, transformationStatus, transformationMatrixH)
    else:
        transformationMatrixH, transformationStatus = None, None

#  Felzenszwalb et al.


def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    out = []
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    x1, x2, y1, y2 = np.array([]), np.array([]), np.array([]), np.array([])

    for box in boxes:
        x1 = np.append(x1, box[0][0])
        y1 = np.append(y1, box[0][1])
        x2 = np.append(x2, box[2][0])
        y2 = np.append(y2, box[2][1])

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:

        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):

            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    for idx in pick:
        out.append(boxes[idx])

    # return only the bounding boxes that were picked
    return out


def image_scanning(
        imageObject,
        imageScene,
        detectionBoxes,
        heightImageObject,
        widthImageObject,
        heightOverlap,
        widthOverlap):

    heightImageScene, widthImageScene = imageScene.shape[:2]

    heightSteps = int(
        math.floor(heightImageScene / (heightImageObject - heightOverlap)))
    widthSteps = int(
        math.floor(widthImageScene / (widthImageObject - widthOverlap)))

    print('Horizontal Steps: %d' % (widthSteps))
    print('Vertical Steps: %d' % (heightSteps))

    colorMap = np.ones((heightImageScene, widthImageScene), np.uint8) * 255
    colorMap = cv2.cvtColor(colorMap, cv2.COLOR_GRAY2BGR)

    for heightStep in range(0, heightSteps - 1):
        for widthStep in range(0, widthSteps - 1):

            iniHeightPosition = int(
                round((heightImageObject - heightOverlap)) * heightStep)
            iniWidthPosition = int(
                round((widthImageObject - widthOverlap)) * widthStep)

            endHeightPosition = iniHeightPosition + heightImageObject
            endWidthPosition = iniWidthPosition + widthImageObject

            if iniHeightPosition < endHeightPosition - 1 and iniWidthPosition < endWidthPosition - 1:

                subImageFromScene = imageScene[iniHeightPosition:endHeightPosition,
                                               iniWidthPosition:endWidthPosition]

                rfactor = 1

                keyPointsImageObject, descriptorsImageObject = detector.detectAndCompute(
                    imageObject, None)
                keyPointsImageScene, descriptorsImageScene = detector.detectAndCompute(
                    subImageFromScene, None)

                MIN_KEYPOINTS = 12
                if len(keyPointsImageScene) > MIN_KEYPOINTS:

                    # Show object image key points
                    vis = np.zeros(
                        (heightImageObject, widthImageObject), np.uint8)
                    vis[:heightImageObject, :widthImageObject] = imageObject
                    kp_color = (250, 0, 0)

                    for marker in keyPointsImageObject:
                        vis = cv2.drawMarker(vis, tuple(int(i)
                                                        for i in marker.pt), color=kp_color)

                    cv2.imshow('key points', vis)

                    # Do matching and draw resuts for this image
                    match_and_draw(colorMap, detectionBoxes, keyPointsImageObject, keyPointsImageScene,
                                   descriptorsImageObject, descriptorsImageScene, iniHeightPosition, iniWidthPosition)

                cv2.waitKey(10)

    return colorMap


def init_detector():
    import sys
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])

    try:
        fn1, fn2 = args
    except:
        fn1 = './a'
        fn2 = './b'

    objectFilename = fn1[fn1.index("/")+1:-4]
    sceneFilename = fn2[fn2.index("/")+1:-4]
    imageObject = cv2.imread(fn1, 0)
    imageScene = cv2.imread(fn2, 0)
    imageOriginal = cv2.imread(fn2)
    detector, matcher = init_feature()

    if imageObject is None:
        print('Failed to load object image:', fn1)
        sys.exit(1)

    if imageScene is None:
        print('Failed to load scene image:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    return imageOriginal, imageObject, imageScene, detector, matcher, objectFilename, sceneFilename


if __name__ == '__main__':

    # Load images and detector
    imageOriginal, imageObject, imageScene, detector, matcher, objectFilename, sceneFilename = init_detector()

    # Add margin around object image to avoid border effect
    MARGIN = 0.1
    heightImageObject, widthImageObject = imageObject.shape[:2]
    marginAroundImageObject = int(heightImageObject * MARGIN / 2)
    imageObjectWithMargin = np.zeros(
        (heightImageObject + marginAroundImageObject * 2, widthImageObject + marginAroundImageObject * 2), np.uint8)
    imageObjectWithMargin[
        marginAroundImageObject: heightImageObject + marginAroundImageObject,
        marginAroundImageObject: widthImageObject + marginAroundImageObject
    ] = imageObject

    # Reuse original image
    imageObject = imageObjectWithMargin
    heightImageObject, widthImageObject = imageObject.shape[:2]

    # Go through the image and perform the search
    # Overlap of 0.8 means that the step_0 is (0,0) and step_1 is (width*0.2, 0)
    detectionBoxes = []
    OVERLAP = 0.25
    heightOverlap = heightImageObject * OVERLAP
    widthOverlap = widthImageObject * OVERLAP
    matchMap = image_scanning(imageObject, imageScene, detectionBoxes, heightImageObject,
                              widthImageObject, heightOverlap, widthOverlap)

    # Group detections
    groupedBoxes = non_max_suppression_slow(detectionBoxes, 0.2)

    for box in detectionBoxes:
        cv2.polylines(matchMap, [box], True, (255, 0, 0), 4)

    for box in groupedBoxes:
        cv2.polylines(imageOriginal, [box], True, (0, 255, 0), 8)

    rfactor = 1

    cv2.imwrite('color_map_' + objectFilename + '_' +
                sceneFilename + '.jpg', matchMap)
    cv2.imwrite('detection_map_' + objectFilename + '_' +
                sceneFilename + '.jpg', imageOriginal)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()
