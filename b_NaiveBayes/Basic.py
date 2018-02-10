#-*- coding: utf-8 -*-
import numpy as np

class NaiveBayes(object):
	"""
		Class initialization
		self._x, self._y: record the variable of training set
	"""
	def __init__(self):
		self._x = self._y = None
		self._data = self._func = None
		self._n_possibilities = None
		self._labeled_x = self._label_zip = None
		self._cat_counter = self._con_counter = None
		self.label_dic = self._feat_dics = None
 
	def __getitem__(self, item):
		if isinstance(item, str):
			return getattr(self, "_"+item)

	def feed_data(self, x, y, sample_weight=None):
		pass

	def feed_sample_weight(self, sample_weight=None):
		pass

	def get_prior_probability(self, lb=1):
		return [(_c_num+lb)/(len(self._y)+lb*len(self._cat_counter)) for _c_num in self._cat_counter]

	def fit(self, x=None, y=None, sample_weight=None, lb=1):
		if x is not None and y is not None:
			self.feed_date(x, y, sample_weight)

		self._func = self._fit(lb)

	def _fit(self, lb):
		pass

	def _transfer_x(self, x):
		pass

	def predict_one(self, x, get_raw_result=False):
		if isinstance(x, np.ndarray):
			x=x.tolist()
		else:
			x = x[:]

		x = self._transfer_x(x)
		m_arg, m_probability = 0, 0

		for i in range(len(self._cat_counter)):
			p = self._func(x, i)
			if p > m_probability:
				m_arg, m_probability = i, p

			if not get_raw_result:
				return self.label_dic[m_arg]

		return m_probability

	def predict(self, x, get_raw_result=False):
		return np.array([self.predict_one(xx, get_raw_result) for xx in x])

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		print("Acc: {}  %".format(100.0*np.sum(y_pred == y)/len(y)))


