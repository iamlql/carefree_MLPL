import numpy as np

class Layer:

	def __init__(self, shape):
		self.shape = shape

	def __str__(self):
		return self.__class__.__name__

	def __repr__(self):
		return str(self)

	@property
	def name(self):
		return str(self)

	def _activate(self, x):
		pass

	def derivative(self, y):
		pass

	def activate(self, x, w, bias):
		return self._activate(x.dot(w) + bias)

	def bp(self, y, w, prev_delta):
		return prev_delta.dot(w.T)*self.derivative(y)

class CostLayer(Layer):
	def __init__(self, shape, cost_function = "MSE"):
		super(CostLayer, self).__init__(shape)
		self._available_cost_function = {
			"MSE": CostLayer._mse, 
			"SVM": CostLayer._svm, 
			"CrossEntropy": CostLayer._cross_entropy
			}

		self._available_transform_functions = {
			"Softmax": CostLayer._softmax,
			"Sigmoid": CostLayer._sigmoid
			}

		self._cost_function_name = cost_function
		self._cost_function = self._available_transform_functions[cost_function]

		if transform is None and cost_function == "CrossEntropy":
			self._transform = "Softmax"
			self._transform_function = CostLayer._softmax
		else:
			self._transform = transform
			self._transform_function = self._available_transform_functions.get(transform, None)

	def __str__(self):
		return self._cost_function_name

	def _activate(self, x, predict):
		if self._transform_function is None:
			return x
		return self._transform_function(x)

	def _derivative(self, y, delta=None):
		pass

	@staticmethod
	def safe_exp(x):
		return np.exp(x - np.max(x, axis = 1, keepdims = True))

	@staticmethod
	def _softmax(y, diff = False):
		if diff:
			return y * (1 - y)
		exp_y = CostLayer.safe_exp(y)
		return exp_y / np.sum(exp_y, axis = 1, keepdims = True)

	@staticmethod
	def _sigmoid(y, diff = False):
		if diff:
			return y * (1 - y)
		return 1 / (1 + np.exp(-y))

	@property
	def calculate(self):
		return lambda y, y_pred: self._cost_function(y, y_pred, False)

	@staticmethod
	def _mse(y, y_pred, diff = True):
		if diff:
			return -y + y_pred
		return 0.5 * np.average((y - y_pred)**2)

	@staticmethod
	def _cross_entropy(y, y_pred, diff = True, eps = 1e-8):
		if diff:
			return -y / (y_pred + eps) + (1 - y) / (1 - y_pred + eps)
		return np.average(-y * np.log(y_pred + eps) - (1-y) * np.log(1 - y_pred + eps))

	def bp_first(self, y, y_pred):
		if self._cost_function_name == "CrossEntropy" and (self._transform == "Softmax" or self._transform == "Sigmoid"):
			return y - y_pred
		dy = -self._cost_function(y, y_pred)
		if self._transform_function is None:
			return dy
		return dy * self._transform_function(y_pred, diff = True)

### Activation Functions ###
class Sigmoid(Layer):
	def _activate(self, x):
		return 1 / (1 + np.exp(-x))

	def derivative(self, y):
		return y * (1 - y)

class Tanh(Layer):
	def _activate(self, x):
		return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

	def derivative(self, y):
		return 1 - y**2

class Relu(Layer):
	def _activate(self, x):
		return max(0, x)

	def derivative(self, y):
		if y == 0:
			return 0
		else:
			return 1

class Elu(Layer):
	def __init__(self, shape, alpha = 1):
		super(Elu, self).__init__(shape)
		self.alpha = alpha

	def _activate(self, x):
		if x >= 0:
			return x
		else:
			return self.alpha * (np.exp(x) - 1)

	def derivative(self, y):
		if y >= 0:
			return 1
		else:
			return y + self.alpha

class Softplus(Layer):
	def _activate(self, x):
		return np.log(1 + np.exp(x))

	def derivative(self, y):
		return 1 - 1 / np.exp(y)

class Identity(Layer):
	def _activate(self, x):
		return x

	def derivative(self, y):
		return 1