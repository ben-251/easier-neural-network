from network import *
from logic import batched

def main():
	main_network = Network((2,10,10,1))
	batch_size = 4
	data_handler = DataHandler()
	samples = data_handler.read_samples()
	for mini_samples in batched(samples, batch_size):
		#TODO: make it write the current id to `current_training_sample.txt` so that it can continue from the right sample
		for sample in mini_samples:
			main_network.setInputLayer(sample.inputs)
			main_network.feedforward()
			main_network.backpropagate()