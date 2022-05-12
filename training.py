from hopfieldnetwork import *
from PIL import Image


def train(hopfieldNetwork, inputPaths, size=120):
    parsedData = []
    for path in inputPaths:
        parsedData.append(image2numpy_array(path, (size, size)).flatten())
    for data in parsedData:
        hopfieldNetwork.train_pattern(data)
    hopfieldNetwork.save_network("networks/network1.npz")
    return parsedData
