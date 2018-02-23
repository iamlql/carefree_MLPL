import numpy as np
from Bases import KernelBase, KernelConfig
from Timing import Timing

class SVM(KernelBase):
	SVMTiming = Timing()
	def __init__(self):
		super(SVM, self).__init__()
		self._fit_args, self._fit_args_names = [1e-3], ['tol']
		self._c =None

	def _pick_first(self, tol):
		con1 = self._alpha > 0
		con2 = self._alpha < self._c

		err1 = self._y * self._prediction_cache - 1
		err2 = err1.copy()
		err3 = err1.copy()

		err1[con1 | (err1 >= 0)] = 0
		err2[(~con1 | ~con2) | (err2 == 0)] = 0
		err3[con2 | (err3 <= 0)] = 0

		err = err1**2 + err2**2 + err3**2
		idx = np.argmax(err)

		if err[idx] < tol:
			return

		return idx

	def _pick_second(self, idx1):
		idx = np.random.randint(len(self._y))
		while idx == idx1:
			idx = np.random.randint(len(self._y))
		return idx

	def _get_lower_bound(self, idx1, idx2):
		if self._y[idx1] != self._y[idx2]:
			return max(0., self._alpha[idx2] - self._alpha[idx1])
		return max(0., self._alpha[idx2] + self._alpha[idx1] - self._c)

	def _get_upper_bound(self, idx1, idx2):
		if self._y[idx1] != self._y[idx2]:
			return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
		return min(self._c, self._alpha[idx1] + self._alpha[idx2])

	def _upper_dw_cache(self, idx1, idx2, da1, da2, y1, y2):
		self._dw_cache = np.array([da1 * y1, da2 * y2])

	def