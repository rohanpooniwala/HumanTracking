from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import dlib
from skimage.measure import structural_similarity as ssim

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
tracked_humans = []

class human_tracker:
    def __init__(self):
        self.label = ""
        self.cam_loc = ""
        self.boundary = None
        self.features = None
        self.object_tracker = dlib.correlation_tracker()
        self.disappeared = True


def get_humans(image):
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return pick

def find_humans_in_camname(cam_name):
    temp = []
    for human in tracked_humans:
        if human.cam_loc == cam_name:
            temp.append(human)
    return temp

def check_overlap(in_boxes, check_boxes):
    x0, y0, x1, y1 = (in_boxes[0][0], in_boxes[0][1], in_boxes[1][0], in_boxes[1][1])
    or_ar = (x1 - x0) * (y1 - y0)                                                           #original area
    for i in range(len(check_boxes)):
        temp_box = check_boxes[i]
        tx0, ty0, tx1, ty1 = (temp_box[0][0], temp_box[0][1], temp_box[1][0], temp_box[1][1])
        #if x0 < tx1 and x1 > tx0 and y0 < ty1 and y1 > ty0:
        #SI = Max(0, Max(XA2, XB2) - Min(XA1, XB1)) * Max(0, Max(YA2, YB2) - Min(YA1, YB1))
        intersection_area = max(0, (min(x1,tx1) - max(x0,tx0)) * (min(y1, ty1) - max(y0,ty0)))
        in_ar = (tx1 - tx0) * (ty1 - ty0)

        overlap = intersection_area / (or_ar + in_ar - intersection_area)

        if overlap > 0.60:
            return i
    return -1

def check_overlap_humans(in_boxes, check_boxes):
    x0, y0, x1, y1 = (in_boxes[0][0], in_boxes[0][1], in_boxes[1][0], in_boxes[1][1])
    or_ar = (x1 - x0) * (y1 - y0)                                                           #original area
    for i in range(len(check_boxes)):
        temp_box = check_boxes[i].boundary
        tx0, ty0, tx1, ty1 = (temp_box[0][0], temp_box[0][1], temp_box[1][0], temp_box[1][1])
        #if x0 < tx1 and x1 > tx0 and y0 < ty1 and y1 > ty0:
        #SI = Max(0, Max(XA2, XB2) - Min(XA1, XB1)) * Max(0, Max(YA2, YB2) - Min(YA1, YB1))
        intersection_area = max(0, (min(x1,tx1) - max(x0,tx0)) * (min(y1, ty1) - max(y0,ty0)))
        in_ar = (tx1 - tx0) * (ty1 - ty0)

        overlap = intersection_area / (or_ar + in_ar - intersection_area)

        if overlap > 0.80:
            return i
    return -1

def match_feature(image, features):
    return False

def crop(image, boundary):
    return None

if __name__ == "__main__":
    vid = cv2.VideoCapture("test1.mp4")
    ret, image = vid.read()
    image = cv2.resize(image, (320, 240))



    cam_name = "cam1"
    count = -1

    while True:
        ret, image = vid.read()
        if ret:
            image = cv2.resize(image, (320, 240))
            detected_humans = get_humans(image)
            detected_humans = [[(xA, yA), (xB, yB)] for (xA, yA, xB, yB) in detected_humans]
            count += 1

            if count%30 == 0:
                already_present_humans = find_humans_in_camname(cam_name)
                if len(detected_humans) > len(already_present_humans):
                    for bounds in already_present_humans:
                        box = bounds.boundary
                        index = check_overlap(box, detected_humans)
                        if index > -1:
                            detected_humans.pop(index)

                    for bounds in detected_humans:
                        #ob_track = dlib.correlation_tracker()
                        #tracker.start_track(img, dlib.rectangle(*points[0]))
                        temp = len(tracked_humans)
                        flag = False
                        for id in range(len(tracked_humans)):
                            human = tracked_humans[id]
                            if human.cam_loc == "":
                                if match_feature(crop(bounds, image), human.features):
                                    temp = id
                                    flag = True

                        if not flag:
                            human = human_tracker()
                            human.label = "Human " + str(temp)
                            tracked_humans.append(human)
                        else:
                            human = tracked_humans[temp]

                        human.cam_loc = cam_name
                        human.bounds = bounds
                        tx0, ty0, tx1, ty1 = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
                        #print(tx0, ty0, tx1, ty1)
                        rect = dlib.rectangle(int(tx0), int(ty0), int(tx1), int(ty1))

                        human.object_tracker.start_track(image, rect)
                        human.disappeared = False

                if len(detected_humans) < len(already_present_humans):
                    for bounds in detected_humans:
                        box = bounds
                        index = check_overlap_humans(box, already_present_humans)
                        if index > -1:
                             already_present_humans[index].cam_loc = ""

            already_present_humans = find_humans_in_camname(cam_name)
            for human in already_present_humans:
                human.object_tracker.update(image)

                # tracker.update(img)
                # cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

                rect = human.object_tracker.get_position()
                pt1 = (int(rect.left()), int(rect.top()))
                pt2 = (int(rect.right()), int(rect.bottom()))
                human.boundary = [pt1, pt2]
                cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, human.label, pt1, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("la", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
