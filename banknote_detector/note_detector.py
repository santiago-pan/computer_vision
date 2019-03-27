import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6


def initAkaze():
    detector = cv2.AKAZE_create()
    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filterMatches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kpPairs = zip(mkp1, mkp2)
    return p1, p2, list(kpPairs)


def exploreMatch(img1, img2, kpPairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(
            corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 0, 0), 4)

    if status is None:
        status = np.ones(len(kpPairs), np.bool_)
    p1, p2 = [], []
    for kpp in kpPairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
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
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow('detection', cv2.resize(vis, None, fx=0.5,
                                       fy=0.5, interpolation=cv2.INTER_CUBIC))

    return vis


def matchAndDraw(imgObject, imgFrame, kp1, desc1, kp2, desc2, matcher):
    rawMatches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    p1, p2, kpPairs = filterMatches(kp1, kp2, rawMatches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found' % len(p1))

    vis = exploreMatch(imgObject, imgFrame, kpPairs, status, H)


def findObj(imgObject, kp1, desc1, imgFrame, detector, matcher):
    kp2, desc2 = detector.detectAndCompute(imgFrame, None)
    matchAndDraw(imgObject, imgFrame, kp1, desc1, kp2, desc2, matcher)


def loadImageObjects():
    imgObject = cv2.imread("50bill_small.jpg", 0)
    return imgObject


def detector():

    # Load objects
    imgObject = loadImageObjects()

    # Init AKAZE detector
    detector, matcher = initAkaze()
    kp1, desc1 = detector.detectAndCompute(imgObject, None)

    cap = cv2.VideoCapture(0)
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        findObj(imgObject, kp1, desc1, frame, detector, matcher)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    cap.release()
    cv2.destroyAllWindows()


detector()
