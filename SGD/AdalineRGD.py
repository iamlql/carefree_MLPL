import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

class AdalineRGD(object):
	def __init__(self, lr = 0.0001, epoch = 10, shuffle = True, random_state = True):
		self.name = self.__class__.__name__
		self.lr = lr
		self.epoch = epoch
		self.w_initialized = False
		self.shuffle = shuffle
		if random_state:
			random.seed()

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __getitem__(self, item):
		if isinstance(item, str):
			return getattr(self, "_"+item)

	def net_input(self, x):
		return np.dot(x, self._w[1:]) + self._w[0]

	def activation(self, x):
		return self.net_input(x)

	def predict(self, x):
		return np.sign(self.activation(x))

	def _initialize_weights(self, m):
		self._w = np.zeros(1+m)
		self.w_initialized = True

	def _shuffle(self, x, y):
		r = np.random.permutation(len(y))
		# print("y: {}".format(len(y)))
		# print("r: {}".format(r))
		return x[r], y[r]

	def partial_fit(self, x, y):
		if not self.w_initialized:
			self._initialize_weights(x.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(x, y):
				self._update_weights(xi, target)
		else:
			self._update_weights(x, y)
		return self

	def fit(self, x, y):
		self._initialize_weights(x.shape[1])
		self._cost = []
		
		for idx in range(self.epoch):
			if self.shuffle:
				x, y = self._shuffle(x, y)
			cost = []
			for xi, target in zip(x, y):
				cost.append(self._update_weights(xi, target))
			avg_cost = sum(cost)/len(y)
			self._cost.append(avg_cost)
		return self

	def _update_weights(self, xi, target):
		output = self.net_input(xi)
		errors = (target - output)
		self._w[1:] += self.lr*xi.T.dot(errors)
		self._w[0]  += self.lr*errors.sum()
		cost = (errors**2).sum() / 2.0
		return cost

class iris_detector(AdalineRGD):
	def __init__(self, lr = 0.01, epoch = 1000, shuffle = True, random_state = True):
		super(AdalineRGD, self).__init__()
		self.lr = lr
		self.epoch = epoch
		self.w_initialized = False
		self.shuffle = shuffle

	@staticmethod
	def get_dataframe(infile):
		df = pd.read_csv(infile, header = None)
		y = df.iloc[0:100, 4].values
		y = np.where(y == 'Iris-setosa', -1, 1)
		x = df.iloc[0:100, [0, 2]].values

		return x, y

	def _fit(self, x, y):
		self.fit(x, y)

		plt.figure()
		plt.plot(range(1, len(self._cost) + 1), self._cost, marker = 'o')
		plt.xlabel('Epochs')
		plt.ylabel('log(SSE)')
		# plt.show()

	def plot_decision_regions(self, x, y, resolution = 0.02):
		
		markers = ('o', 'x' , 's', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
		labels = ('setosa', 'versicolor')
		cmap = ListedColormap(colors[:len(np.unique(y))])

		x1_min, x1_max = x[:, 0].min()  - 1, x[:, 0].max()  + 1
		x2_min, x2_max = x[:, 1].min()  - 1, x[:, 1].max()  + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
		self._fit(x, y)
		z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		z = z.reshape(xx1.shape)
		plt.figure()
		plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		for idx, c1 in enumerate(np.unique(y)):
			plt.scatter(x[y == c1, 0], x[y == c1, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = labels[idx])
		plt.xlabel('petal length')
		plt.ylabel('sepal length')
		plt.legend(loc = 'upper left')
		plt.show()


if __name__ == "__main__":
	id = iris_detector(0.01, 20)
	x, y = iris_detector.get_dataframe('./data/iris.data')
	# id.plot_decision_regions(x, y)

	x_std = np.copy(x)
	x_std[:,0] = (x[:, 0] - x[:,0].mean()) / x[:,0].std()
	x_std[:,1] = (x[:, 1] - x[:,1].mean()) / x[:,1].std()
	id.plot_decision_regions(x_std, y)