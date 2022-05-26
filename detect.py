import cv2 as cv
import numpy
from hopfieldnetwork import *
import numpy as np
from PIL import Image

def convert(input, n):
    return input.flatten()


f_ind = 0


def detect(hopfieldNetwork, training_data, _input, n):
    _input = np.where(_input <= 0, 1, -1)
    im = Image.fromarray(_input)
    im.save("detect.png")
    hopfieldNetwork.set_initial_neurons_state(convert(_input, n))
    hopfieldNetwork.update_neurons(iterations=2, mode="async")
    mini = float('inf')
    min_idx = -1
    global f_ind
    for idx, data in enumerate(training_data):
        # print(sum(data))
        im = np.copy(data)
        im = numpy.reshape(im, (120, 120))
        # print("image", im)
        # im = Image.fromarray(im)
        # im.save("{}.png".format(f_ind))
        f_ind += 1
        im = Image.fromarray(hopfieldNetwork.S)
        im.save("hopfield inside.png")
        hd = hamming_distance(data, hopfieldNetwork.S)
        if hd < mini:
            min_idx = idx
            mini = hd
        print(hd, " ", idx)
    return min_idx
