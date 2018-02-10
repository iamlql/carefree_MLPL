import numpy as np

class ClassfierBase:
	def __init__(self, *args, **kwargs):
		self.name = self.__class__.__name__

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __getitem__(self, item):
		if isinstance(item, str):
			return getattr(self, "_"+item)

	@staticmethod
	def acc(y, y_pred):
		y = np.array(y)
		y_pred = np.array(y_pred)
		return 1.0*np.sum(y == y_pred)/len(y)

	def predict(self, x, get_raw_results = False):
		pass

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		y = np.array(y)
		print ClassfierBase.acc(y, y_pred)
