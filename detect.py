import cv2 as cv
import numpy
from hopfieldnetwork import *
import numpy as np


def convert(input, n):
    return input.flatten()

f_ind =0
def detect(hopfieldNetwork, training_data, _input, n):
    hopfieldNetwork.set_initial_neurons_state(convert(_input, n))
    hopfieldNetwork.update_neurons(iterations=2, mode="async")
    mini = float('inf')
    min_idx = -1
    global f_ind
    for idx, data in enumerate(training_data):
        print(sum(data))
        im = data
        im = numpy.reshape(data,(120,120))
        cv.imwrite("data/"+str(f_ind)+".jpg",im)
        f_ind +=1
        hd =hamming_distance(data, hopfieldNetwork.S)
        if hd < mini:
            min_idx = idx
            mini =hd
        print(hd," ", idx)
    return min_idx
