def hough(img):
    lines = cv2.HoughLines(img, 1, np.pi/(180), 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3200*(-b))
            y1 = int(y0 + 3200*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(bird_img, (x1, y1), (x2, y2), (0, 0, 255), 6)
