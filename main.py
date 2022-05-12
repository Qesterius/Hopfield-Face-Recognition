from hopfieldnetwork import HopfieldNetwork
#https://github.com/andreasfelix/hopfieldnetwork?fbclid=IwAR0cO-9uE_S9jOso5VNpSQPUaN7mv0J74XDjgm4NtJNuKjDiwXGUQU1vzVU


hopfield_network1 = HopfieldNetwork(N=100)

#TODO:load images to input pattern
input_pattern = None
#train
hopfield_network1.train_pattern(input_pattern)
#update neurons
hopfield_network1.update_neurons(iterations=5, mode="async")
#Compute the energy function of a pattern:
hopfield_network1.compute_energy(input_pattern)
#Save a network as a file:
hopfield_network1.save_network("path/to/file")

#Open an already trained Hopfield network:
#hopfield_network2 = HopfieldNetwork(filepath="network2.npz")
