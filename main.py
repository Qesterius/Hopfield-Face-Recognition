from __future__ import division, print_function
import numpy as np
import os, sys
from time import process_time

sys.path.append("../../..")
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi, plot_network_development, DATA_DIR
import pathlib
from PIL import Image

timer = process_time()

### Hopfield Modells mit 10000 Neuronen: Bilder bekannter Physiker
print("4.0 Test des Hopfield Modells")
print("Hopfield Modells mit 10000 Neuronen: Bilder bekannter Physiker")
# Lade Bilder bekannter Physiker als NumPy array
N = 100 ** 2
path_list = [
    os.path.join(pathlib.Path().resolve(), "data", f)
    for f in [
        "einstein.jpg",
    ]
]
xi = images2xi(path_list, N)
# Speichere/Trainiere Bilder bekannter Physiker im Netzwerk
hopfield_network = HopfieldNetwork(N=N)
hopfield_network.train_pattern(xi)

## Test: Bildrekonstruktion aus Teilbild
# Setze 'halben' Einstein als Startkonfiguartion
einstein = np.copy(xi[:, 0])
half_einstein = np.copy(xi[:, 0])
half_einstein[: int(N / 2)] = -1
hopfield_network.set_initial_neurons_state(np.copy(half_einstein))
# Plotte Neuronenkonfiguartion fuer 3 Zeitschritte
plot_network_development(
    hopfield_network, 3, "async", einstein, os.path.join(pathlib.Path().resolve(), "reconstruct_einstein.pdf")
)

### Oszillationen im synchronen Modus
print("\n\n4.0 Oszillationen im synchronen Modus")
hopfield_network = HopfieldNetwork(N=4)
pattern1 = np.array([1, 1, -1, -1])
pattern2 = np.array([1, -1, 1, -1])
patterns = np.column_stack((pattern1, pattern2))
hopfield_network.train_pattern(patterns)
hopfield_network.set_initial_neurons_state(np.array([1, -1, -1, -1]))

plot_network_development(
    hopfield_network,
    6,
    "sync",
    pattern1,
    "sync_oscillation.pdf",
    anno_hamming=False,
)

print("\nProcess time: {:.3f} s".format(process_time() - timer))