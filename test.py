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

einstein = image2numpy_array("data/einstein.jpg", (120, 120)).flatten()
dirac = image2numpy_array("data/dirac.jpg", (120, 120)).flatten()
de_broile = image2numpy_array("data/de_broglie.jpg", (120, 120)).flatten()
# curie = image2numpy_array("data/curie.jpg", (120, 120)).flatten()

input = [einstein, dirac, de_broile]

half_dirac = np.copy(dirac)
half_dirac[: int(N / 2)] = -1


hopfield_network1 = HopfieldNetwork(N=N)

for i in input:
    hopfield_network1.train_pattern(i)



hopfield_network1.set_initial_neurons_state(half_dirac)

# data0 = Image.fromarray(np.reshape(hopfield_network1.S, (n, n)), mode='L')
# data0.save("inside.png")
# print(hopfield_network1.S)

# data0 = Image.fromarray(half_einstein.reshape((n, n)), mode='L')
# data0.save("half.png")

hopfield_network1.update_neurons(iterations=5, mode="async")

print(hopfield_network1.S)
print(np.sum(hopfield_network1.S))
print(hamming_distance(hopfield_network1.S, de_broile))

data = Image.fromarray(np.reshape(hopfield_network1.S, (n, n)), mode='L')
data.save("result.png")
