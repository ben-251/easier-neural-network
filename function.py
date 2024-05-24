from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

Number = Union[float, np.float_, np.ndarray]

class Function():
	def __init__(self):
		...
	
	def __call__(self, value: Number):
		return self.__call__(value)

class DerivableFunction(ABC, Function):
	'''
	Base method for most functions.
	Not to be instantiated directly
	'''
	derivative: Function

	def __init__(self):
		super().__init__()
		self.derivative = Function()
		self.set_default_derivative()


	@abstractmethod
	def set_default_derivative(self):
		self.derivative.__call__ = lambda value: np.where(value > 0, 1, 0)
	
	@abstractmethod
	def computeDerivative(self, wrt):
		...

	@abstractmethod
	def __call__(self, value:Number):
		return self.__call__(value)		

class Relu(DerivableFunction):
	def __init__(self):
		super().__init__()
	
	def __call__(self,x): 
		return np.maximum(0.0, x, dtype=np.dtype("float64"))
	
	def computeDerivative(self, wrt):
		raise NotImplementedError

	def set_default_derivative(self):
		self.derivative.__call__ = lambda value: np.where(value > 0, 1, 0)
