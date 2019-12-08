import cv2
vidcap = cv2.VideoCapture('assets/videoplayback.mp4')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
