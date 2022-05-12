import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cascPath = "haarcascade_frontalface_default.xml"


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
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)"""


def trimPhotoToRect(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    cropped = img[x:x + w, y:y + h]
    return cropped


def drawFaceRect(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img
plt.ion()

""" kurwa chcialem stestoewac funkcje powyzsze, ale nie chce mi sie kamerka zalaczyc"""
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
screen = plt.imshow(img)
plt.show()

while (True):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()
    screen.set_data(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    plt.pause(0.2)
    plt.draw()

    if cv.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        screen.close()
        break


# cap.release()
