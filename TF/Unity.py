import numpy as np
import tensorflow as tf
import os

class DataUtil:
	@staticmethod
	def read_csv(batch_size, file_name, record_defaults):
		filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)+'/'+file_name])
		reader = tf.TextLineReader(skip_header_lines = 1)
		key, value = reader.read(filename_queue)

		decoded = tf.decode_csv(value, record_defaults = record_defaults)
		return tf.train.shuffle_batch(decoded, batch_size = batch_size, capacity = batch_size*50, min_after_dequeue = batch_size)

	@staticmethod
	def get_dataset(batch_size, file_name, record_defaults, train_num=None, tar_idx = None, shuffle=True, quantize=False, one_hot=False, **kwargs):
		x = []
		with open(file_name, "r", encoding="utf8") as file:
			for sample in file:
				str_list = sample.strip().split(",")
				if str_list != ['']:
					x.append(str_list)

		if shuffle:
			np.random.shuffle(x)

		tar_idx = -1 if tar_idx is None else tar_idx
		if record_defaults[tar_idx] == [""]:
			y = [xx.pop(tar_idx) for xx in x]
		else:
			y = np.array([xx.pop(tar_idx) for xx in x], dtype = np.int8)
			if one_hot:
				y = (y[..., None] == np.arange(np.max(y) + 1))
    
		if [""] not in record_defaults.pop(tar_idx):
			x = np.array(x, dtype = np.float32)

		if not quantize:
			if train_num is None:
				return x, y
			else:
				return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])
		else:
			x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, **kwargs)
			if one_hot:
				y = (y[..., None] == np.arange(np.max(y)+1)).astype(np.int8)

			if train_num is None:
				return x, y, wc, features, feat_dicts, label_dict
			else:
				return ((x[:train_num], y[:train_num]), (x[train_num:], y[train_num:]), wc, features, feat_dicts, label_dict)

	@staticmethod
	def quantize_data(x, y, wc=None, continuous_rate=0.1, separate=False):
		if isinstance(x, list):
			xt = map(list, zip(*x))
		else:
			xt = x.T
		features = [set(feat) for feat in xt]
		if wc is None:
			wc = np.array([len(feat) >= int(continuous_rate * len(y)) for feat in features])
		elif not wc:
			wc = np.array([False] * len(xt))
		else:
			wc = np.asarray(wc)
		feat_dicts = [{_l: i for i, _l in enumerate(feats)} if not wc[i] else None for i, feats in enumerate(features)]
		if not separate:
			if np.all(~wc):
				dtype = np.int
			else:
				dtype = np.float32
			x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)] for sample in x], dtype=dtype)
		else:
			x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)] for sample in x], dtype=np.float32)
			x = (x[:, ~wc].astype(np.int), x[:, wc])
		label_dict = {l: i for i, l in enumerate(set(y))}
		y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
		label_dict = {i: l for l, i in label_dict.items()}
		return x, y, wc, features, feat_dicts, label_dict

	@staticmethod
	def gen_xor(size=100, scale=1, one_hot=True):
		x = np.random.randn(size) * scale
		y = np.random.randn(size) * scale
		z = np.zeros((size, 2))
		z[x * y >= 0, :] = [0, 1]
		z[x * y < 0, :] = [1, 0]
		if one_hot:
			return np.c_[x, y].astype(np.float32), z
		return np.c_[x, y].astype(np.float32), np.argmax(z, axis=1)

	@staticmethod
	def gen_spiral(size=50, n=7, n_class=7, scale=4, one_hot=True):
		xs = np.zeros((size * n, 2), dtype=np.float32)
		ys = np.zeros(size * n, dtype=np.int8)
		for i in range(n):
			ix = range(size * i, size * (i + 1))
			r = np.linspace(0.0, 1, size+1)[1:]
			t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
			xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
			ys[ix] = i % n_class
		if not one_hot:
			return xs, ys
		return xs, np.array(ys[..., None] == np.arange(n_class), dtype=np.int8)

	@staticmethod
	def gen_random(size=100, n_dim=2, n_class=2, scale=1, one_hot=True):
		xs = np.random.randn(size, n_dim).astype(np.float32) * scale
		ys = np.random.randint(n_class, size=size).astype(np.int8)
		if not one_hot:
			return xs, ys
		return xs, np.array(ys[..., None] == np.arange(n_class), dtype=np.int8)

	@staticmethod
	def gen_two_clusters(size=100, n_dim=2, center=0, dis=2, scale=1, one_hot=True):
		center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
		center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
		cluster1 = (np.random.randn(size, n_dim) + center1) * scale
		cluster2 = (np.random.randn(size, n_dim) + center2) * scale
		data = np.vstack((cluster1, cluster2)).astype(np.float32)
		labels = np.array([1] * size + [0] * size)
		indices = np.random.permutation(size * 2)
		data, labels = data[indices], labels[indices]
		if not one_hot:
			return data, labels
		labels = np.array([[0, 1] if label == 1 else [1, 0] for label in labels], dtype=np.int8)
		return data, labels

	@staticmethod
	def gen_noisy_linear(size=10000, n_dim=100, n_valid=5, noise_scale=0.5):
		x_train = np.random.randn(size, n_dim)
		x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
		x_test = np.random.randn(int(size*0.15), n_dim)
		idx = np.random.permutation(n_dim)[:n_valid]
		w = np.random.randn(n_valid, 1)
		y_train = (x_train[..., idx].dot(w) > 0).astype(np.float32).ravel()
		y_test = (x_test[..., idx].dot(w) > 0).astype(np.float32).ravel()
		return (x_train_noise, y_train), (x_test, y_test)



	