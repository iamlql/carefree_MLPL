import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
	def __init__(self, lr = 0.01, epoch = 10):
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

	def fit(self, x, y):
		self._w = np.zeros(1+x.shape[1])
		self._errors = []

		for _ in range(self.epoch):
			errors = 0
			for xi, target in zip(x, y):
				update = self.lr * (target - self.predict(xi))
				self._w[1:] += update * xi
				self._w[0]  += update
				errors += int(update != 0.0)
			self._errors.append(errors)
		return self

	def net_input(self, x):
		return np.dot(x, self._w[1:]) + self._w[0]

	def predict(self, x):
		# return np.where(self.net_input(x) >= 0.0, 1, -1)
		return np.sign(self.net_input(x))

	@staticmethod
	def acc(y, y_pred):
		y = np.array(y)
		y_pred = np.array(y_pred)
		return 1.0*np.sum(y == y_pred)/len(y)

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		y = np.array(y)
		print (Perceptron.acc(y, y_pred))

class iris_detector(Perceptron):
	def __init__(self, lr = 0.01, epoch = 10):
		super(Perceptron, self).__init__()
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
		# x, y = iris_detector.get_dataframe('./data/iris.data')
		self.fit(x, y)

		plt.figure()
		plt.plot(range(1, len(self._errors) + 1), self._errors, marker = 'o')
		plt.xlabel('Epochs')
		plt.ylabel('Number of misclassifications')
		# plt.show()

	def plot_decision_regions(self, resolution = 0.02):
		x, y = iris_detector.get_dataframe('./data/iris.data')
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
	id = iris_detector()
	id.plot_decision_regions()