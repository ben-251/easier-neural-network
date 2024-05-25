from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

Number = Union[float, np.float_, np.ndarray]

class BaseFunction():
	def __init__(self):
		...
	
	def __call__(self, value: Number):
		return self.__call__(value)

class CostFunction(ABC):
	derivative: BaseFunction

	def __init__(self):
		super().__init__()
		self.set_default_derivative()
	
	@abstractmethod
	def __call__(self,actual_activation, expected_activation): 
		return self.__call__(actual_activation, expected_activation)

	@abstractmethod
	def set_default_derivative(self):
		raise NotImplementedError

	@abstractmethod
	def differentiate(self, wrt):
		#TODO: find a way to encode W vs a vs b so i can do cost wrt w, cost wrt a, etc.
		...

class DerivableFunction(ABC, BaseFunction):
	'''
	Base method for most functions.
	Not to be instantiated directly
	'''
	derivative: BaseFunction

	def __init__(self):
		super().__init__()
		self.derivative = BaseFunction()
		self.set_default_derivative()

	@abstractmethod
	def set_default_derivative(self):
		self.derivative.__call__ = lambda value: np.where(value > 0, 1, 0)
	
	@abstractmethod
	def differentiate(self, wrt):
		'''
			Sets the derivative of the method
		'''
		...

	@abstractmethod
	def __call__(self, value:Number):
		return self.__call__(value)

class MSECost(CostFunction):
	def __init__(self):
		super().__init__()
		self.set_default_derivative()

	def __call__(self,actual_activation, expected_activation): 
		return (actual_activation - expected_activation)**2
	
	def set_default_derivative(self):
		self.derivative.__call__ = lambda actual_activation,\
		expected_activation: 2*(actual_activation - expected_activation)
		#TODO: change behaviour of derivatives to not have to be the layer by layer kind s
		# so i can reuse it wherever I need a derivative. OR make two kinds of functions: 
		# the regular functions used throughout the program, and the cost function with its 
		# two params

	def differentiate(self):
		raise NotImplementedError("not applicable to Cost Functions!")	

class Relu(DerivableFunction):
	def __init__(self):
		super().__init__()
	
	def __call__(self,x): 
		return np.maximum(0.0, x, dtype=np.dtype("float64"))
	
	def differentiate(self, wrt):
		raise NotImplementedError

	def set_default_derivative(self):
		self.derivative.__call__ = lambda value: np.where(value > 0, 1, 0)




	