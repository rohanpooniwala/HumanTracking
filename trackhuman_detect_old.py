from __future__ import print_function
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
from skimage.measure import structural_similarity as ssim

objects = []

threshold = 0.35

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('test1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
counter = 0
while cap.isOpened():
    ret, image = cap.read()

    image = imutils.resize(image, width=min(400, image.shape[1]))
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    flag = 0
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
