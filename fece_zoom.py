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

""" kurwa chcialem stestoewac funkcje powyzsze, ale nie chce mi sie kamerka zalaczyc"""
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
img = frame
screen = plt.imshow(img)
plt.show()
goodFrame = False
while (True):
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    faceRects =getFaces(frame)
    faces =[]
    for f in faceRects:
        faces.append(cv.resize(trimPhotoToRect(frame, f), (120, 120)))
        drawFaceRect(frame,f)



    # gogoog ML here to frame if goodframe==true
    for face_id in range(len(faces)):
        out="aa"
        cv.putText(frame,out,(faceRects[face_id][0],faceRects[face_id][1]),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0, 0))
        #frame = faces[face_id]



    screen.set_data(frame)
    plt.pause(0.2)
    plt.draw()
    goodFrame = False
    if cv.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        screen.close()
        break


# cap.release()
