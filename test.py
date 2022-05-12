import hopfieldnetwork
import numpy as np
from hopfieldnetwork import *
from PIL import Image
import matplotlib.pyplot as plt


def convert(x):
    if x: return 1
    return -1


vconv = np.vectorize(convert)

n = 120
N = n ** 2
width = n
height = n

# grayscale image
# img = Image.open("data/einstein.jpg").convert('L')
# img = img.resize((width, height), Image.ANTIALIAS)

# input_pattern = Image.open("data/einstein.jpg")
# input_pattern = input_pattern.convert("1")
# input_pattern = input_pattern.resize((100, 100))

# input_pattern = np.array(input_pattern)

input_pattern1 = image2numpy_array("data/einstein.jpg", (120, 120)).flatten()
input_pattern2 = image2numpy_array("data/dirac.jpg", (120, 120)).flatten()
input_pattern3 = image2numpy_array("data/de_broglie.jpg", (120, 120)).flatten()





hopfield_network1 = HopfieldNetwork(N=N)

hopfield_network1.train_pattern([])

half_einstein = np.copy(input_pattern)
half_einstein[: int(N / 2)] = -1

hopfield_network1.set_initial_neurons_state(np.copy(half_einstein))

data0 = Image.fromarray(np.reshape(input_pattern, (n, n)), mode='L')
data0.save("input.png")
print(hopfield_network1.S)

# data0 = Image.fromarray(np.reshape(hopfield_network1.S, (n, n)), mode='L')
# data0.save("inside.png")
# print(hopfield_network1.S)

# data0 = Image.fromarray(half_einstein.reshape((n, n)), mode='L')
# data0.save("half.png")

hopfield_network1.update_neurons(iterations=5, mode="async")

print(hopfield_network1.S)
print(np.sum(hopfield_network1.S))

data = Image.fromarray(np.reshape(hopfield_network1.S, (n, n)), mode='L')

data.save("result.png")
