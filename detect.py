import cv2 as cv
from hopfieldnetwork import *
import numpy as np


def convert(input, n):
    return image2numpy_array(input, (n, n)).flatten()


def detect(hopfieldNetwork, training_data, input, n):
    hopfieldNetwork.set_initial_neurons_state(convert(input, n))
    hopfieldNetwork.update_neurons(iterations=5, mode="async")
    mini = float('inf')
    min_idx = -1
    for idx, data in enumerate(training_data):
        if hamming_distance(data, hopfieldNetwork.S) < mini:
            min_idx = idx
    return min_idx
