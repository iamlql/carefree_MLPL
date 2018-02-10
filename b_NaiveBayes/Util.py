#-*- coding: utf-8 -*-
import numpy as np

class DataUtil:
	def get_dataset(self, name, path, train_num = None, tar_idx = None, shuffle = True):
		# print path
		x = list()
		with open(path, "r") as file:    #, encoding="utf8"
			if "balloon" in name:
				for sample in file:
					x.append(sample.strip().split(","))

		if shuffle:
			np.random.shuffle(x)

		tar_idx = -1 if tar_idx is None else tar_idx
		y = np.array([xx.pop(tar_idx) for xx in x])
		x = np.array(x)

		if train_num is None:
			return x, y
		return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])