import time

import cv2
import cv2 as cv
import matplotlib.pyplot as plt

from hopfieldnetwork import *
import cv2 as cv
from hopfieldnetwork import HopfieldNetwork
from os import listdir
from os.path import isfile, join
from training import train
from detect import detect
import numpy

load = True
cascPath = "haarcascade_frontalface_default.xml"


def init():
    hopfieldNet = []
    size = 0
    trained = []
    if not load:
        size = 120
        directory = ["data/lukasz", "data/kamil", "data/mateusz"]
        hopfieldNet = HopfieldNetwork(size ** 2)
        pathList = [join(dire, f) for dire in directory for inde, f in enumerate(listdir(dire)) if
                    isfile(join(dire, f)) and inde < 1]
        trained = train(hopfieldNet, pathList, size)
        numpy.save("networks/data2.npy", trained)
        numpy.save("networks/size2.npy", size)
        hopfieldNet.save_network("networks/network2.npz")
    else:
        hopfieldNet = HopfieldNetwork(filepath="networks/network2.npz")
        size = numpy.load("networks/size2.npy")
        trained = numpy.load("networks/data2.npy")

    return trained, hopfieldNet, size


def getFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces
    """
    displaying faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """


def trimPhotoToRect(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    cropped = img[y:y + h, x:x + w]
    return cropped


def drawFaceRect(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


def matchInterpret(match):
    if match == 0:
        return "Kamil"
    if match == 1:
        return "Lukasz"
    if match == 2:
        return "Mateusz"


plt.ion()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
img = frame
fig, ax = plt.subplots(1, 3)
for a in ax:
    a.set_axis_off()
screen = ax[0].imshow(img)
plt.show()
goodFrame = False

trainingData, hopfieldNetwork, size = init()
prev_time = 0
frame_rate = 1
while True:
    ret, frame = cap.read()
    print(type(frame))
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()
    time_elapsed = time.time() - prev_time
    print("time", time_elapsed)

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        faceRects = getFaces(frame)
        faces = []
        frame2 = []
        frame3 = []

        for f in faceRects:
            _p = cv.resize(trimPhotoToRect(frame, f), (120, 120))
            _p = cv.cvtColor(_p, cv.COLOR_BGR2GRAY)
            img = Image.fromarray(_p, 'L')
            img = img.convert("1")
            frame2 = img
            img.save("camera.png")
            faces.append(np.array(img))
            print(img)
            drawFaceRect(frame, f)

        # gogoog ML here to frame if goodframe==true
        for face_id in range(len(faces)):
            match = detect(hopfieldNetwork, trainingData, faces[face_id], size)
            out = matchInterpret(match)
            frame3 = np.reshape(trainingData[match], (120, 120))
            print("matched with label: ", matchInterpret(match))
            cv.putText(frame, str(out), (faceRects[face_id][0], faceRects[face_id][1]), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       1,
                       (255, 0, 0))
            # frame = faces[face_id]

        ax[0].imshow(frame)
        if type(frame2) == type(Image):
            ax[1].imshow(frame2)
        if len(frame3) != 0:
            ax[2].imshow(frame3)
        plt.pause(0.1)
        plt.draw()
        goodFrame = False
        if cv.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            plt.close()
            break

# cap.release()
