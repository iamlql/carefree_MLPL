import numpy as np
from matplotlib.colors import ListedColormap

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