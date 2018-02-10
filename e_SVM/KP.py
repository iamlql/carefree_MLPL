import numpy as np
from Bases import KernelBase

class KernelPerceptron(KernelBase):
	"""docstring for KernelPerceptron"""
	def __init__(self):
		super(KernelPerceptron, self).__init__()
		self._fit_args, self._fit_args_names = [1], ['lr']

	def _update_dw_cache(self, idx, lr, sample_weight):
		self._dw_cache = lr * self._y[idx] * sample_weight[idx]

	def _update_db_cache(self, idx, lr, sample_weight):
		self._db_cache = self._dw_cache

	def _update_params(self):
		self._w = self._alpha * self._y
		self._b = np.sum(self._w)

	def _fit(self, sample_weight, lr):
		err = (np.sign(self._prediction_cache) != self._y) * sample_weight
		indices = np.random.permutation(len(self._y))
		idx = indices[np.argmax(err[indices])]
		if self._prediction_cache[idx] == self._y[idx]:
			return True
		self._alpha[idx] += lr
		self._update_dw_cache(idx, lr, sample_weight)
		self._update_db_cache(idx, lr, sample_weight)
		self._update_pred_cache(idx)