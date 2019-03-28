import cv2
import numpy as np
from classes import BankNote


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


def exploreMatch(imgObject, imgFrame, kpPairs, status=None, H=None):
    downSamplingFactor = 2
    imgToShow = imgObject[::downSamplingFactor, ::downSamplingFactor]
    h1, w1 = imgObject.shape[:2]
    h2, w2 = imgFrame.shape[:2]
    hs, ws = imgToShow.shape[:2]

    vis = np.zeros((h2, w2), np.uint8)

    if status is not None and np.sum(status) > 20:
        vis[:h2, :w2] = imgFrame
        vis[:hs, :ws] = imgToShow
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32(cv2.perspectiveTransform(
                corners.reshape(1, -1, 2), H).reshape(-1, 2))
            cv2.polylines(vis, [corners], True, (255, 0, 0), 4)

        if status is None:
            status = np.ones(len(kpPairs), np.bool_)
        p1, p2 = [], []

        for kpp in kpPairs:
            p1.append(np.int32(kpp[0].pt))
            p2.append(np.int32(np.array(kpp[1].pt)))

        green = (0, 255, 0)
        red = (0, 0, 255)
        factor = 2
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):

            x1 = int(x1 / downSamplingFactor)
            y1 = int(y1 / downSamplingFactor)

            if inlier:
                # col = green
                cv2.circle(vis, (x1, y1), 2, green, -1)
                cv2.circle(vis, (x2, y2), 2, green, -1)
            else:
                # col = red
                r = 2
                thickness = 3

                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), red, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), red, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), red, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), red, thickness)
        # vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            x1 = int(x1 / downSamplingFactor)
            y1 = int(y1 / downSamplingFactor)
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

    else:
        vis[:h2, :w2] = imgFrame
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    return vis


def matchAndDraw(imgObject, imgFrame, kp1, desc1, kp2, desc2, matcher):
    rawMatches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kpPairs = filterMatches(kp1, kp2, rawMatches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, None

    vis = exploreMatch(imgObject, imgFrame, kpPairs, status, H)
    return vis, status


def findObj(imgObject, kp1, desc1, imgFrame, detector, matcher):
    kp2, desc2 = detector.detectAndCompute(imgFrame, None)
    return matchAndDraw(imgObject, imgFrame, kp1, desc1, kp2, desc2, matcher)


def loadBankNoteImage(imgFile, detector):
    bankNote = BankNote()
    bankNote.img = cv2.imread('./images/' + imgFile, 0)
    bankNote.kp, bankNote.desc = detector.detectAndCompute(bankNote.img, None)
    return bankNote


def loadBankNotes(detector):
    imageFiles = ["50bill_small.jpg",
                  "5pounds_small.jpg",
                  "100czechkoruna_small.jpg"]
    return [loadBankNoteImage(img, detector) for img in imageFiles]


def detector(captureVideo):

    # Capture video
    videOut = None
    if captureVideo:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videOut = cv2.VideoWriter('output.avi', fourcc,
                                  20.0, (1280, 720))

    # Init AKAZE detector
    detector, matcher = initAkaze()

    # Load objects
    bankNotes = loadBankNotes(detector)

    # Start video
    cap = cv2.VideoCapture(0)
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        winnerIndex = 0
        winnerScore = 0
        for i in range(0, len(bankNotes)):
            bankNotes[i].vis, bankNotes[i].status = findObj(bankNotes[i].img,
                                                            bankNotes[i].kp,
                                                            bankNotes[i].desc,
                                                            frame, detector,
                                                            matcher)
            currentScore = np.sum(bankNotes[i].status)

            if bankNotes[i].status is not None and currentScore > winnerScore:
                winnerScore = currentScore
                winnerIndex = i

        cv2.imshow('detection', cv2.resize(bankNotes[winnerIndex].vis, None,
                                           fx=0.5,
                                           fy=0.5,
                                           interpolation=cv2.INTER_CUBIC))

        if captureVideo:
            videOut.write(bankNotes[winnerIndex].vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    cap.release()
    cv2.destroyAllWindows()

    if captureVideo:
        videOut.release()


detector(False)
