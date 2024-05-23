import numpy as np
import bentests as bt
from layer import *
from network import Network
from DataHandler import *
from logic import *

class Ignore(bt.testGroup):
	...

class Maths(bt.testGroup):
	def testRelu(self):
		startingMatrix = np.array([[1, -2], [3, 0]])
		relu = Relu()
		result = relu.main(startingMatrix) #hmmm not sure what to do
		bt.assertEquals(result, np.array([[1, 0], [3, 0]])) 

	def testReluPrime(self):
		startingMatrix = np.array([[0.3, 0], [-0.9, -45], [0, 23]])
		relu = Relu()
		result = relu.derivative(startingMatrix)
		bt.assertEquals(result, np.array([[1,0], [0, 0], [0, 1]]))

	def testToBitZero(self):
		n = to_bit(0.1)
		bt.assertEquals(n, 0.0)

	def testToBitHalfway(self):
		n = to_bit(0.5)
		bt.assertEquals(n, 1)

	def testCostFunction(self):
		cost_function = CostFunction()
		cost = cost_function.main(0.2,0.5)
		bt.assertAlmostEquals(cost, 0.09)

	def testCostFunctionDerivative(self):
		cost_function = CostFunction()
		cost_derivative = cost_function.derivative(0.2,0.5)
		bt.assertAlmostEquals(cost_derivative, -0.6)

class Layers(bt.testGroup):
	def testLayerSizeGiven(self):
		new_layer = Layer(layer_size=3)
		bt.assertEquals(new_layer.size,3)

	def testDefaultLayerSize(self):
		new_layer = Layer()
		bt.assertEquals(new_layer.size,10)

	def testDefaultWeightedLayerSize(self):
		new_layer = WeightedLayer(prevLayer=Layer())
		bt.assertEquals(new_layer.size,10)

	def testWeightedLayerShape(self):
		previous_layer = Layer(layer_size=15)
		main_layer = WeightedLayer(previous_layer,layer_size=10)
		bt.assertEquals(main_layer.weights.shape, (10,15))

	def testWeightedLayerBiasShape(self):
		previous_layer = Layer(layer_size=4)
		main_layer = WeightedLayer(previous_layer, layer_size=10)
		bt.assertEquals(main_layer.biases.shape, (10,1))

	def testLayerActivations(self):
		layer = InputLayer(layer_size=2)
		bt.assertEquals(layer.activations, np.array([[0.0],[0.0]]))

	def testWeightedLayerActivations(self):
		input_layer = InputLayer(layer_size=3)
		first_hidden_layer = WeightedLayer(prevLayer=input_layer, layer_size=3)
		bt.assertEquals(first_hidden_layer.activations, np.array([[0.0],[0.0],[0.0]]))

	def testWeightedLayerWeights(self):
		input_layer = InputLayer(layer_size=2)
		first_hidden_layer = WeightedLayer(prevLayer=input_layer, layer_size=4)
		bt.assertEquals(
			first_hidden_layer.weights,
			np.array(
				[[0.0, 0.0], [0.0 , 0.0], [0.0, 0.0], [0.0, 0.0]]
			)
		)

	def test_feed_forward_with_zero_weights(self):
		network = Network((2,3,4))
		data_handler = DataHandler()
		sample = data_handler.read_samples()[0]
		network.setInputLayer(sample.inputs)
		network.feedforward()
		bt.assertEquals(network.layers[1].activations, np.array([[0.0],[0.0],[0.0]]))

class NetworkTests(bt.testGroup):
	def testInputLayerActivationTypes(self):
		network = Network((1,2))
		bt.assertEquals(network.layers[0].activations, np.array([[0.0]]))

	def testHiddenLayerActivationTypes(self):
		network = Network((1,2))
		bt.assertEquals(network.layers[1].activations, np.array([[0.0], [0.0]]))

	def testHiddenLayerWeights(self):
		network = Network((3,2))
		bt.assertEquals(
			network.layers[1].weights,
			np.array([
				[0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0]
			])
		)

	def testFeedForwardZeroed(self):
		network = Network((2,3,4,1))
		data_handler = DataHandler()
		sample = data_handler.read_samples()[0]
		network.setInputLayer(sample.inputs)
		network.feedforward()
		bt.assertEquals(
			network.layers[-2].activations,
			np.array([[0.0], [0.0], [0.0], [0.0]])
		) # cuz if w and b is 0, aw+b = 0 for all a

	def testFeedForwardTypeNotZeroed(self):
		network = Network((2,3,4,1),customWeightValue=0.25,customBiasValue=0.7)
		network.setInputLayer([0.3, 0.1])
		network.feedforward()
		bt.assertEquals(
			network.layers[1].activations.dtype,
			np.array([[0.8], [0.8], [0.8]]).dtype
		) # all neurons end up the same cuz the weights and biases are the same

	def testFeedForwardNotZeroed(self):
		network = Network((2,3,4,1),customWeightValue=0.25,customBiasValue=0.7)
		network.setInputLayer([0.3, 0.1])
		network.feedforward()
		bt.assertAlmostEquals(
			network.layers[1].activations,
			np.array([[0.8], [0.8], [0.8]])
		) # all neurons end up the same cuz the weights and biases are the same

	def testInputLayer(self):
		network = Network((2,3,4))
		data_handler = DataHandler()
		sample = data_handler.read_samples()[0]
		network.setInputLayer(sample.inputs)
		bt.assertEquals(network.layers[0].activations, np.array(
			[[0.8410888466588518],[0.1267111713120581]] # i need a better method for testing this part than just copy pasting the first two items..
		))

	def testStoreBiases(self):
		network = Network((2,5,1))
		network.storeWeightsAndBiases()
		new_network = Network((2,5,1))
		new_network.loadWeightsAndBiases()
		bt.assertEquals(
			new_network.layers[1].biases,
			np.array([[0.0],[0.0],[0.0],[0.0],[0.0]])
		)

	def testStoreWeights(self):
		network = Network((2,5,1),customWeightValue=0.4)
		network.storeWeightsAndBiases()
		new_network = Network((2,5,1))
		new_network.loadWeightsAndBiases()
		bt.assertEquals(
			new_network.layers[1].weights,
			np.array([[0.4, 0.4],[0.4, 0.4],[0.4, 0.4],[0.4, 0.4],[0.4, 0.4]])
		)

# class DataHandlerTests(bt.testGroup):
# 	'''
# 	DON'T RUN IF YOU KEEP THE TEST DATA UNCHANGED
# 	'''
# 	def testSampleInputType(self):
# 		data_handler = DataHandler(training_size=10,testing_size=10)
# 		data_handler.create_data()
# 		data_handler.write_data()
# 		samples = data_handler.read_samples()
# 		bt.assertEquals(type(samples[0].inputs), tuple)
	
# 	def testSampleOutputType(self):
# 		data_handler = DataHandler(training_size=10,testing_size=10)
# 		data_handler.create_data()
# 		data_handler.write_data()
# 		samples = data_handler.read_samples()
# 		bt.assertEquals(type(samples[0].result), float)		

# 	def testSampleOutputRange(self):
# 		data_handler = DataHandler(training_size=10,testing_size=10)
# 		data_handler.create_data()
# 		data_handler.write_data()
# 		samples = data_handler.read_samples()
# 		bt.assertEquals(samples[0].result == 0.0 or samples[0].result == 1.0, True)

bt.test_all(
	Maths,
	Layers,
	NetworkTests,
)
