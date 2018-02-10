import numpy as np

class KernelConfig:
	default_c = 1
	default_p = 3

class KernelBase(object):
	def __init__(self):
		self._fit_args,self._fit_args_names = None, []
		self._x = self._y = self._gram = None
		self._w = self._b = self._alpha = None
		self._kernel = self._kernel_name = self._kernel_param = None
		self._prediction_cache = self._dw_cache = self._db_cache = None

	@staticmethod
	def _poly(x, y, p):
		return (x.dot(y.T) + 1) ** p

	@staticmethod
	def _rbf(x, y, gamma):
		return np.exp(-gamma * np.sum((x[..., None, :]-y) ** 2, axis = 2))

	def _update_dw_cache(self, *args):
		pass

	def _update_db_cache(self, *args):
		pass

	def fit(self, x, y, kernel = 'rbf', epoch = 10 ** 4, ** kwargs):
		if kernel == "poly":
			_p = kwargs.get("p", KernelConfig.default_p)
			self._kernel_name = "Polynomial"
			self._kernel_param = "degree = {}".format(_p)
			self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
		elif kernel == "rbf":
			_gamma = kwargs.get("gamma", 1 / self._x.shape[1])
			self._kernel_name = "RBF"
			self._kernel_param = r"$\gamma = {:8.6}$".format(_gamma)
			self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)
		else:
			raise NotImplementedError("Kernel '{}' has not defined".format(kernel))

		self._alpha, self._w, self._prediction_cache = (np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)))
		self._gram = self._kernel(self._x, self._x)
		self._b = 0

		fit_args = []
		for name, arg in zip(self._fit_args_names, self._fit_args):
			if name in kwargs:
				arg = kwargs[name]
			fit_args.append(arg)

		for i in range(epoch):
			if self._fit(sample_weight, *fit_args):
				break

		self._update_params()

	def _prepare(self, sample_weight, **kwargs):
		pass

	def _fit(self, *args):
		pass

	def _update_params(self):
		pass

	def _update_pred_cache(self, *args):
		self._prediction_cache += self._db_cache
		if len(args) == 1:
			self._prediction_cache += self._dw_cache * self._gram[args[0]]
		else:
			self._prediction_cache += self._dw_cache.dot(self._gram[args, ...])

	def predict(self, x, get_raw_results=False, gram_provided=False):
		if not gram_provided:
			x = self._kernel(self._x, np.atleast_2d(x))
		y_pred = self._w.dot(x) + self._b
		if not get_raw_results:
			return np.sign(y_pred)
		return y_pred

