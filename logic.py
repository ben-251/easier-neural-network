from typing import Callable, overload
from typing_extensions import overload
import numpy as np
from itertools import islice

class Function():
	def __init__(self):
		self.main = self.get_main_function()
		self.derivative = self.get_derivative()
	
	def get_main_function(self):
		raise NotImplementedError("Cannot call on base class")

	def get_derivative(self, with_respect_to) -> Callable[[float], float]:
		...

class Relu(Function):
	def __init__(self):
		super().__init__()
	
	def get_main_function(self): 
		return lambda x: np.maximum(0.0, x, dtype=np.dtype("float64"))

	def get_derivative(self):
		return lambda Array: np.where(Array > 0, 1, 0)

class CostFunction(Function):
	def __init__(self):
		super().__init__()
	
	def get_main_function(self):
		return lambda actual_activation, expected_activation: (actual_activation - expected_activation)**2
	
	def get_derivative(self):
		return lambda actual_activation, expected_activation: 2*(actual_activation - expected_activation)

def to_bit(n:float) -> bool:
	if n < 0 or n > 1:
		raise ValueError("n must be from 0 - 1")
	elif n >= 0.5:
		return True
	else: 
		return False

def xor(a:float,b:float) -> float:
	a = to_bit(a)
	b = to_bit(b)
	if (a and not b) or (b and not a):
		return 1.0
	else:
		return 0.0

def batched(iterable, n):
	if n < 1:
		raise ValueError('n must be at least one')
	it = iter(iterable)
	while batch := tuple(islice(it, n)):
		yield batch