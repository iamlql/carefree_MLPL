import numpy as np
# from Bases import ClassifierBase

class Perceptron():
	def __init__(self):
		# super(Perceptron, self).__init__()
		self._w = self._b = 0

	def fit(self, x, y, sample_weight = None, lr = 0.01, epoch = 10**6):
		# x, y = np.alleast_2d(x), np.array(y)
		if sample_weight is None:
			sample_weight = np.ones(len(y))
		else:
			sample_weight = np.array(sample_weight)*len(y)
		# print x.shape
		self._w = np.zeros(x.shape[1])
		self._b = 100
		y_pred  = np.zeros(y.shape)
		for _ in xrange(epoch):
			print 'Step: ', _
			y_pred = self.predict(x)
			print("Acc: {}  %".format(100.0*np.sum(y_pred == y)/len(y)))
			_err = (y_pred != y) * sample_weight

			#SGD
			_indices = np.random.permutation(len(y))
			_idx = _indices[np.argmax(_err[_indices])]
			if y_pred[_idx] == y[_idx]:
				return
			_delta = lr*y[_idx]*sample_weight[_idx]
			self._w += _delta*x[_idx]
			self._b += _delta

	def svm_fit(self, x, y, C = 10, sample_weight = None, lr = 0.01, epoch = 10**6):
		# x, y = np.alleast_2d(x), np.array(y)
		if sample_weight is None:
			sample_weight = np.ones(len(y))
		else:
			sample_weight = np.array(sample_weight)*len(y)
		# print x.shape
		self._w = np.zeros(x.shape[1])
		self._b = 100
		y_pred  = np.zeros(y.shape)
		for _ in xrange(epoch):
			print 'Step: ', _
			y_pred = self.predict(x, get_raw_result=True)
			print("Acc: {}  %".format(100.0*np.sum(self.predict(x) == y)/len(y)))
			_err = 1-y*y_pred
			_indices = np.random.permutation(len(y))
			_idx = _indices[np.argmax(_err[_indices])]
			if _err[_idx] <= 0:
				return
			_delta = lr*C*y[_idx]*sample_weight[_idx]
			self._w = (1-lr)*self._w+_delta*x[_idx]
			self._b += _delta

	def predict(self, x, get_raw_result = False):
		# print "w:", self._w
		# print "b:", self._b
		rs = np.sum(self._w*x, axis=1)+self._b
		if not get_raw_result:
			return np.sign(rs)
		return rs

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		print("Acc: {}  %".format(100.0*np.sum(y_pred == y)/len(y)))

def main():
	total_num = 100
	training_num = 60
	# test_num = total_num - 60
	import gen_data
	xyz = gen_data.gen_linspace(50, 100, size = total_num)
	test_perceptron = Perceptron()
	test_perceptron.svm_fit(xyz[:training_num,:-1],xyz[:training_num,-1],5)
	test_perceptron.evaluate(xyz[training_num:,:-1],xyz[training_num:,2])


if __name__ == '__main__':
	main()
