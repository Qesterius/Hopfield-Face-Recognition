import cv2 as cv
from hopfieldnetwork import HopfieldNetwork
import numpy as np


def getInput(n):
    hopfieldnetwork = HopfieldNetwork(filepath="path/to/file")
    cap = cv.VideoCapture(n)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv.imshow('frame', frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thres = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        hopfieldnetwork.set_initial_neurons_state(np.copy(np.array(gray)))
        hopfieldnetwork.update_neurons(iterations=5, mode="async")
        hopfieldnetwork.compute_energy(hopfieldnetwork.S)
        cv.imshow('res', hopfieldnetwork.S)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv.destroyAllWindows()