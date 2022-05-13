from hopfieldnetwork import *
from PIL import Image
from tqdm import tqdm
import numpy

def train(hopfieldNetwork, inputPaths, size=120):
    parsedData = []
    for path in inputPaths:
        parsedData.append(image2numpy_array(path, (size, size)).flatten())
    ar = numpy.array(parsedData)
    print(ar.shape)
    hopfieldNetwork.train_pattern(ar.transpose())

    hopfieldNetwork.save_network("networks/network1.npz")
    return parsedData
