import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
	def __init__(self, lr = 0.0001, epoch = 10):
		self.name = self.__class__.__name__
		self.lr = lr
		self.epoch = epoch

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

	def fit(self, x, y):
		self._w = np.zeros(1+x.shape[1])
		self._cost = []

		for idx in range(self.epoch):
			output = self.net_input(x)
			errors = (y - output)
			self._w[1:] += self.lr*x.T.dot(errors)
			self._w[0]  += self.lr*errors.sum()
			cost = (errors**2).sum() / 2.0
			self._cost.append(cost)
		return self

class iris_detector(AdalineGD):
	def __init__(self, lr = 0.0001, epoch = 1000):
		super(AdalineGD, self).__init__()
		self.lr = lr
		self.epoch = epoch

	@staticmethod
	def get_dataframe(infile):
		df = pd.read_csv(infile, header = None)
		y = df.iloc[0:100, 4].values
		y = np.where(y == 'Iris-setosa', -1, 1)
		x = df.iloc[0:100, [0, 2]].values

		return x, y

	def _fit(self, x, y):
		# x_std = np.copy(x)
		# x_std[:,0] = (x[:, 0] - x[:,0].mean()) / x[:,0].std()
		# x_std[:,1] = (x[:, 1] - x[:,1].mean()) / x[:,1].std()
		# self.fit(x_std, y)
		self.fit(x, y)

		plt.figure()
		plt.plot(range(1, len(self._cost) + 1), np.log10(self._cost), marker = 'o')
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
	id = iris_detector(0.01, 15)
	x, y = iris_detector.get_dataframe('./data/iris.data')
	# id.plot_decision_regions(x, y)

	x_std = np.copy(x)
	x_std[:,0] = (x[:, 0] - x[:,0].mean()) / x[:,0].std()
	x_std[:,1] = (x[:, 1] - x[:,1].mean()) / x[:,1].std()
	id.plot_decision_regions(x_std, y)