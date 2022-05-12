import numpy as np
from hopfieldnetwork import *
from PIL import Image

def convert_to_int(input):
    n = len(input)
    output = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if input[i, j]: output[i, j] = 1
            else: output[i, j] = -1
    return output

n = 100
N = 100 ** 2
width = 100
height = 100

# grayscale image
# img = Image.open("data/einstein.jpg").convert('L')
# img = img.resize((width, height), Image.ANTIALIAS)

input_pattern = Image.open("data/einstein.jpg").convert('L')
input_pattern = input_pattern.convert("1")
input_pattern = input_pattern.resize((100, 100))

input_pattern = np.array(input_pattern)
# input_pattern = convert_to_int(input_pattern)
input_pattern = input_pattern.flatten()


print(input_pattern.shape)


hopfield_network1 = HopfieldNetwork(N=N)
print(hopfield_network1.S)

hopfield_network1.train_pattern(input_pattern)

half_einstein = np.copy(input_pattern)
half_einstein[: int(N / 2)] = -1


hopfield_network1.set_initial_neurons_state(np.copy(half_einstein))
data0 = Image.fromarray(np.reshape(input_pattern, (100, 100)), mode='L')
data0.save("input.png")
print(hopfield_network1.S)

data0 = Image.fromarray(half_einstein.reshape((100, 100)), mode='L')
data0.save("half.png")

hopfield_network1.update_neurons(iterations=5, mode="async")

print(hopfield_network1.S)
print(np.sum(hopfield_network1.S))

data = Image.fromarray(np.reshape(hopfield_network1.S, (100, 100)), mode='L')

data.save("result.png")
