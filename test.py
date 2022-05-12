import numpy as np
from hopfieldnetwork import HopfieldNetwork
from PIL import Image

N = 100
width = 100
height = 100
# grayscale image
img = Image.open("data/einstein.jpg").convert('L')
img = img.resize((width, height), Image.ANTIALIAS)

input_pattern = np.array(img)

hopfield_network1 = HopfieldNetwork(N=N**2)
hopfield_network1.train_pattern(input_pattern)


half_einstein = img[: int(N / 2)] = -1


hopfield_network1.set_initial_neurons_state(half_einstein)

hopfield_network1.update_neurons(iterations=5, mode="async")

hopfield_network1.compute_energy()