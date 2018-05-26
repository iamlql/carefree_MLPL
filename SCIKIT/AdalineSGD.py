from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def plot_decision_regions(x, y, classifier, test_idx = None, resolution = 0.02):
	markers = ('o', 'x' , 's', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
	labels = ('setosa', 'versicolor')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	x1_min, x1_max = x[:, 0].min()  - 1, x[:, 0].max()  + 1
	x2_min, x2_max = x[:, 1].min()  - 1, x[:, 1].max()  + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	z = z.reshape(xx1.shape)
	plt.figure()
	plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	x_test, y_test = x[test_idx,:], y[test_idx]
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(x[y == c1, 0], x[y == c1, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = c1)

	if test_idx:
		x_test, y_test = x[test_idx,:], y[test_idx]
		plt.scatter(x_test[:,0], x_test[:,1], c = '', alpha = 1.0, linewidth = 1, marker = 'o', s = 55, label = 'test set')

def perceptron_visualization(x_train_std, y_train, x_test_std, y_test):
	ppn = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 0)
	ppn.fit(x_train_std, y_train)
	y_pred = ppn.predict(x_test_std)
	x_combined_std = np.vstack((x_train_std, x_test_std))
	y_combined     = np.hstack((y_train, y_test))
	plot_decision_regions(x = x_combined_std, y = y_combined, classifier = ppn, test_idx = range(105, 150))
	plt.xlabel('petal length')
	plt.ylabel('petal width')
	plt.legend(loc = 'upper left')
	plt.show()

def logistic_regression_visualization(x_train_std, y_train, x_test_std, y_test):
	lr = LogisticRegression(C=1000, random_state = 0)
	lr.fit(x_train_std, y_train)
	x_combined_std = np.vstack((x_train_std, x_test_std))
	y_combined     = np.hstack((y_train, y_test))
	plot_decision_regions(x = x_combined_std, y = y_combined, classifier = lr, test_idx = range(105, 150))
	plt.xlabel('petal length')
	plt.ylabel('petal width')
	plt.legend(loc = 'upper left')
	plt.show()

def svm_visualizaion(x_train_std, y_train, x_test_std, y_test):
	svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
	svm = SVC(kernel = 'rbf', C = 1.0, gamma = 0.2, random_state = 0)
	svm.fit(x_train_std, y_train)
	x_combined_std = np.vstack((x_train_std, x_test_std))
	y_combined     = np.hstack((y_train, y_test))
	plot_decision_regions(x = x_combined_std, y = y_combined, classifier = svm, test_idx = range(105, 150))
	plt.xlabel('petal length')
	plt.ylabel('petal width')
	plt.legend(loc = 'upper left')
	plt.show()

if __name__ == "__main__":
	iris = datasets.load_iris()
	x = iris.data[:, [2,3]]
	y = iris.target

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
	sc = StandardScaler()
	sc.fit(x_train)
	x_train_std = sc.transform(x_train)
	x_test_std = sc.transform(x_test)

	weights, params = [], []
	for c in np.arange(-6, 10, dtype = float):
		lr = LogisticRegression(C = (10**c), random_state = 0)
		lr.fit(x_train_std, y_train)
		weights.append(lr.coef_[1])
		params.append(10**c)

	weights = np.array(weights)
	plt.plot(params, weights[:,0], label = 'petal length')
	plt.plot(params, weights[:,1], linestyle = '--', label = 'petal width')
	plt.xlabel('C')
	plt.ylabel('weight coefficient')
	plt.legend(loc = 'upper left')
	plt.xscale('log')
	plt.show()
