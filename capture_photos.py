import sys
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

cascPath = "haarcascade_frontalface_default.xml"
sFactor = 1.1
mNeighbors = 5

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





def getFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cascPath)
    global sFactor
    global mNeighbors
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=sFactor,
        minNeighbors=mNeighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


modes = ["lukasz", "kamil", "mateusz"]
mode = modes[2]


def savePhotos(photos):
    for count, value in enumerate(photos):
        cv.imwrite("data/" + mode + "/" + str(count) + ".png", value)
        print("" + mode + "/" + str(count) + ".png")


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

startCapturing = False


def on_press(event):
    sys.stdout.flush()
    global sFactor
    global mNeighbors
    global startCapturing
    if event.key.lower() == 'o':
        sFactor -= 0.1
        print("sFactor = ", sFactor)
    if event.key.lower() == 'p':
        sFactor += 0.1
        print("sFactor = ", sFactor)
    if event.key.lower() == 'k':
        mNeighbors -= 0.1
        print("sFactor = ", mNeighbors)
    if event.key.lower() == 'l':
        mNeighbors += 0.1
        print("sFactor = ", mNeighbors)
    if event.key.lower() == 'c':
        startCapturing = True
        print("saving")
    if event.key.lower() == 'q':
        savePhotos(gemby)
        cv2.destroyAllWindows()
        screen.close()
        exit(0)


fig, _dupa = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
gemby = []
while (True):
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    faceRects = getFaces(frame)
    faces = []
    for f in faceRects:
        faces.append(cv.resize(trimPhotoToRect(frame, f), (120, 120)))
        drawFaceRect(frame, f)

    # gogoog ML here to frame if goodframe==true
    for face_id in range(len(faces)):
        out = mode
        cv.putText(frame, out, (faceRects[face_id][0], faceRects[face_id][1]), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   (255, 0, 0))
        # frame = faces[face_id]
        if startCapturing:
            gemby.append(faces[face_id])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    screen.set_data(frame)
    plt.pause(0.2)
    plt.draw()
    goodFrame = False
