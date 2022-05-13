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
    hopfieldNet=[]
    size=0
    trained=[]
    if not load:
        size = 120
        directory = ["data/lukasz","data/kamil","data/mateusz"]
        hopfieldNet = HopfieldNetwork(size**2)
        pathList = [join(dire, f) for dire in directory for inde, f in enumerate(listdir(dire)) if isfile(join(dire, f)) and inde < 4]
        trained = train(hopfieldNet, pathList, size)
        numpy.save("networks/data2.npy",trained)
        numpy.save("networks/size2.npy",size)
        hopfieldNet.save_network("networks/network2.npz")
    else :
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
    cropped = img[ y:y + h,x:x + w]
    return cropped


def drawFaceRect(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img



plt.ion()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
img = frame
screen = plt.imshow(img)
plt.show()
goodFrame = False

trainingData, hopfieldNetwork, size = init()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    faceRects = getFaces(frame)
    faces =[]
    print(type(frame))
    for f in faceRects:
        _p = cv.resize(trimPhotoToRect(frame, f), (120, 120))
        _p = cv.cvtColor(_p, cv.COLOR_BGR2GRAY)
        _ret, tresh = cv.threshold(_p,50,255,cv.THRESH_BINARY)
        faces.append(tresh)
        drawFaceRect(frame,f)



    # gogoog ML here to frame if goodframe==true
    for face_id in range(len(faces)):
        match = detect(hopfieldNetwork, trainingData, faces[face_id], size)
        out=match

        print("matched with label: ", match)
        cv.putText(frame,str(out),(faceRects[face_id][0],faceRects[face_id][1]),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0, 0))
        #frame = faces[face_id]

    screen.set_data(frame)
    plt.pause(0.1)
    plt.draw()
    goodFrame = False
    if cv.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        screen.close()
        break


# cap.release()
