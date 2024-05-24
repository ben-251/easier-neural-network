import numpy as np
from DataHandler import DataHandler
import function as func
# from logic import *

class Layer:
	def __init__(self, layer_size: int | None = None):
		if layer_size is None:
			layer_size = 10
		self.size: int = layer_size
		self.activations: np.ndarray = np.zeros((layer_size,1))
	
	def __len__(self):
		return self.activations.shape[0]

class WeightedLayer(Layer):	
	'''
	The class for all non-input layers.
	Formed of weights (matrix), activations (vector), and biases (vector)
	
	activations are updated in the typical fashion.
	'''
	def __init__(self,prevLayer: Layer, layer_size: int | None = None, customWeightValue: float |None = None, customBiasValue: float | None = None):
		# remember its not x,y but "number of rows x number of columns"
		super().__init__(layer_size=layer_size)
		if customWeightValue is None:
			customWeightValue = 0.0 # defaults to zero for now
		if customBiasValue is None:
			customBiasValue = 0.0
		self.weights: np.ndarray = np.zeros((len(self), len(prevLayer))) + customWeightValue #TODO: make Weights and biases randomised
		self.biases: np.ndarray = np.zeros((len(self), 1)) + customBiasValue
	
	def updateActivations(self, previous_layer: Layer):
		relu = func.Relu()
		self.activations = relu(
			np.dot(self.weights, previous_layer.activations) + self.biases
		)
	
class InputLayer(Layer):
	def __init__(self, layer_size: int | None = None):
		super().__init__(layer_size = layer_size)
	
		