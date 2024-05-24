class Parent():
	def __init__(self):
		...

	def __call__(self, n):
		return self.__call__(n)

class Child():
	def __init__(self) -> None:
		self.a = Parent()
		self.a.__call__ = lambda n: n+1

child = Child()
print(child.a(1))