import random
from typing import Dict, Tuple, List
from logic import *
from enum import Enum
import json
from json import JSONEncoder
import numpy as np

class Sample:
	'''
	Individual test samples, for example: 
	inputs = (0.234, 0.892)
	result = 1.0
	'''
	def __init__(self, inputs: List[float]|float, result: float):
		if isinstance(inputs,float):
			self.inputs = [inputs]
		else:
			self.inputs = inputs
		self.result = result

class DataType(Enum):
	TESTING = "data/test_data.json"
	TRAINING = "data/training_data.json"

class DataHandler:
	'''
	Primary tool for creating, writing, and reading data for training and testing the model.
	'''
	def __init__(self,training_size:int|None=None, testing_size: int|None = None) -> None:
		if training_size is None:
			training_size = 60_000
		if testing_size is None:
			testing_size = 10_000
		self.training_size = training_size
		self.testing_size = testing_size

	def create_data(self) -> None:
		self.training_data = self.generate_samples(self.training_size)
		self.testing_data = self.generate_samples(self.testing_size)	
	
	def generate_samples(self, n) -> List[Sample]:
		samples: List[Sample] = []
		for _ in range(n):
			m = 2.0
			c = 3.0
			x = random.uniform(-5,5) # previously 5*random.random()*(-1)**random.choice([0,-1]) lol
			y = m*x + c
			samples.append(Sample(x,y))
		return samples
	
	def write_data(self) -> None:
		self.write_samples(self.training_data, isTraining=True)
		self.write_samples(self.testing_data, isTraining=False)
	
	def write_samples(self, data: List[Sample], isTraining:bool|None=None) -> None:
		data_type = DataType.TRAINING if isTraining else DataType.TESTING
		data_as_json = self.data_to_json(data)
		with open(data_type.value, "w") as f:
			f.write(data_as_json)


	def data_to_json(self, data: List[Sample]) -> str:
		samples = []
		for i, sample in enumerate(data):
			sample_as_dict = {
				"id": i,
				"inputs": [],
				"output": 0.0
			}

			for input_ in sample.inputs: #type: ignore REALLY WEIRD THIS ONE
				sample_as_dict["inputs"].append(input_)

			sample_as_dict["output"] = sample.result
			samples.append(sample_as_dict)
		return json.dumps(samples,indent=4)

	def read_samples(self, isTesting: bool|None=None) -> List[Sample]:
		'''
		Default value of isTesting is effectively false
		'''
		samples = []
		data_type = DataType.TESTING if isTesting else DataType.TRAINING
		with open(data_type.value, "r") as f:
			data = json.load(f)
			for data_sample in data:
				samples.append(Sample(data_sample["inputs"], data_sample["output"]))
		return samples

	def encode_matrix(self, matrix) -> str:
		numpyData = {"array": matrix}
		return json.dumps(numpyData, cls=NumpyArrayEncoder)

	def decode_matrix(self,json_object) -> np.ndarray:
		decodedArrays = json.loads(json_object)
		return np.asarray(decodedArrays["array"])
		
class NumpyArrayEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self, obj)


