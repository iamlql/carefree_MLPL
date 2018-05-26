import numpy as np
import cv2
from Timing import Timing
import matplotlib.pyplot as plt


class KernelConfig:
	default_c = 1
	default_p = 3

class TimingBase:
    def show_timing_log(self):
        pass


class ModelBase:
    """
        Base for all models
        Magic methods:
            1) __str__     : return self.name; __repr__ = __str__
            2) __getitem__ : access to protected members 
        Properties:
            1) name  : name of this model, self.__class__.__name__ or self._name
            2) title : used in matplotlib (plt.title())
        Static method:
            1) disable_timing  : disable Timing()
            2) show_timing_log : show Timing() records
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        self._plot_label_dict = {}
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }
        self._params = {
            "sample_weight": kwargs.get("sample_weight", None)
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    @property
    def title(self):
        return str(self) if self._title is None else self._title

    @staticmethod
    def disable_timing():
        ModelBase.clf_timing.disable()

    @staticmethod
    def show_timing_log(level=2):
        ModelBase.clf_timing.show_timing_log(level)

    # Handle animation

    @staticmethod
    def _refresh_animation_params(animation_params):
        animation_params["show"] = animation_params.get("show", False)
        animation_params["mp4"] = animation_params.get("mp4", False)
        animation_params["period"] = animation_params.get("period", 1)

    def _get_animation_params(self, animation_params):
        if animation_params is None:
            animation_params = self._params["animation_params"]
        else:
            ClassifierBase._refresh_animation_params(animation_params)
        show, mp4, period = animation_params["show"], animation_params["mp4"], animation_params["period"]
        return show or mp4, show, mp4, period, animation_params

    def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                          name=None, img=None):
        if draw_ani and x.shape[1] == 2 and (i + 1) % ani_period == 0:
            if img is None:
                img = self.get_2d_plot(x, y, **animation_params)
            if name is None:
                name = str(self)
            if show_ani:
                cv2.imshow(name, img)
                cv2.waitKey(1)
            if make_mp4:
                ims.append(img)

    def _handle_mp4(self, ims, animation_properties, name=None):
        if name is None:
            name = str(self)
        if animation_properties[2] and ims:
            VisUtil.make_mp4(ims, name)

    def get_2d_plot(self, x, y, padding=1, dense=200, draw_background=False, emphasize=None, extra=None, **kwargs):
        pass

    # Visualization

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        indices = [labels == i for i in range(np.max(labels) + 1)]
        scatters = []
        plt.figure()
        plt.title(title)
        for idx in indices:
            scatters.append(plt.scatter(axis[0][idx], axis[1][idx], c=colors[idx]))
        plt.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                   ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        indices = [labels == i for i in range(n_label)]
        scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in indices:
            scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                  ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.show()

    # Util

    def predict(self, x, get_raw_results=False, **kwargs):
        pass

class ClassfierBase(object):
	def __init__(self, *args, **kwargs):
		self.name = self.__class__.__name__

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __getitem__(self, item):
		if isinstance(item, str):
			return getattr(self, "_"+item)

	@staticmethod
	def acc(y, y_pred):
		y = np.array(y)
		y_pred = np.array(y_pred)
		return 1.0*np.sum(y == y_pred)/len(y)

	def predict(self, x, get_raw_results = False):
		pass

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		y = np.array(y)
		print (ClassfierBase.acc(y, y_pred))

class KernelBase(ClassfierBase):
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
		self._x, self._y = np.atleast_2d(x), np.array(y)
		if kernel == "poly":
			_p = kwargs.get("p", KernelConfig.default_p)
			self._kernel_name = "Polynomial"
			self._kernel_param = "degree = {}".format(_p)
			self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
		elif kernel == "rbf":
			_gamma = kwargs.get("gamma", 1.0 / self._x.shape[1])
			self._kernel_name = "RBF"
			self._kernel_param = "gamma = {}".format(_gamma)
			self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)
		else:
			raise NotImplementedError("Kernel '{}' has not defined".format(kernel))

		self._alpha, self._w, self._prediction_cache = (np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)))
		self._gram = self._kernel(self._x, self._x)
		self._b = 0
		self._prepare(**kwargs)

		fit_args = []
		for name, arg in zip(self._fit_args_names, self._fit_args):
			# if name in kwargs:
			# 	arg = kwargs[name]
			fit_args.append(arg)

		for i in range(epoch):
			if self._fit(1, *fit_args):
				break

		self._update_params()

	def _prepare(self, sample_weight, **kwargs):
		pass

	def _fit(self, sample_weight, *fit_args):
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

	def evaluate(self, x, y):
		y_pred = self.predict(x)
		print("Acc: {}  %".format(100.0*np.sum(y_pred == y)/len(y)))

