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
cap = cv2.VideoCapture('./random people walk studies.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
counter = 0
while(cap.isOpened()):
    ret, image = cap.read()

    image = imutils.resize(image, width=min(400, image.shape[1]))
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    flag = 0
    for (xA, yA, xB, yB) in pick:
        roi = image[yA:yB, xA:xB]
        roi = cv2.resize(roi,(2*40, 2*80))
        roi_path = '/home/sourav/Work/SIH/Tracking/KNN/data/'+str(counter)+'.jpg'
        counter += 1
        cv2.imwrite(roi_path,roi)
        if len(objects) == 0:
            objects.append(roi)
            break
        for i in range(0,len(objects)):

            chans_roi = cv2.split(roi)
            chans_objects = cv2.split(objects[i])
            dis_r = ssim(chans_roi[0], chans_objects[0])
            dis_g = ssim(chans_roi[1], chans_objects[1])
            dis_b = ssim(chans_roi[2], chans_objects[2])

            if dis_r > threshold and dis_g > threshold and dis_b > threshold :
                objects[i] = roi
                font = cv2.FONT_HERSHEY_SIMPLEX
                flag = 1
                cv2.putText(image,str(i),(xA,yA), font, 1,(255,255,255),2,cv2.LINE_AA)
                break
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        if flag == 0:
            objects.append(roi)

    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
