from hopfieldnetwork import *
from PIL import Image
from tqdm import tqdm
import numpy

def image2numpy_array1(path, size):
    img_pil = Image.open(path)
    img_pil = img_pil.resize(size)
    img_pil = img_pil.convert("1")  # convert image to black and white
    img_pil.save("training.png")
    img_np = np.zeros(img_pil.size, dtype="uint8")
    for i in range(img_pil.size[0]):
        for j in range(img_pil.size[1]):
            img_np[i, j] = img_pil.getpixel((i, j))
    # img_np = np.where(img_np <= 0, 1, -1)
    return img_np.transpose()

def train(hopfieldNetwork, inputPaths, size=120):
    parsedData = []
    for path in inputPaths:
        parsedData.append(image2numpy_array1(path, (size, size)).flatten())

    ar = numpy.array(parsedData)
    print(ar.shape)
    # hopfieldNetwork.train_pattern(ar.transpose())
    im = Image.fromarray(ar)
    # im.save("training.png")

    # hopfieldNetwork.save_network("networks/network1.npz")
    return parsedData

