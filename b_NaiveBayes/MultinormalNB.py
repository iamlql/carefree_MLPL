#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from Basic import *

class MultinomialNB(NaiveBayes):
	"""

	"""
	def feed_date(self, x, y, sample_weight=None):

		if isinstance(x, list):
			xt = map(list, zip(*x))  
		else:
			xt = x.T

		features = [set(feat) for feat in xt]
		
		feat_dics = [{_l:i for i,_l in enumerate(feats)} for feats in features]

		label_dic = {_l:i for i, _l in enumerate(set(y))}

		x = np.array([[feat_dics[i][_l] for i,_l in enumerate(sample)] for sample in x])
		y = np.array([label_dic[yy] for yy in y], dtype=np.int8)

		cat_counter = np.bincount(y)

		n_possibilities = [len(feats) for feats in features]

		labels = [y == value for value in xrange(len(cat_counter))]

		labelled_x = [x[ci].T for ci in labels]
		
		self._x, self._y = x, y
		self._labelled_x, self.label_zip = labelled_x, list(zip(labels, labelled_x))
		(self._cat_counter, self._feat_dics, self._n_possibilities) = (cat_counter, feat_dics, n_possibilities)

		self.label_dic = {i: _l for _l, i in label_dic.items()}
		self.feed_sample_weight(sample_weight)

	def feed_sample_weight(self, sample_weight=None):
		self._con_counter = list()
		for dim, _p in enumerate(self._n_possibilities):
			if sample_weight is None:
				self._con_counter.append([np.bincount(xx[dim], minlength = _p) for xx in self._labelled_x])

			else:
				self._con_counter.append([np.bincount(xx[dim],weights=sample_weight[label]/sample_weight[label].mean(), minlength=_p) for label, xx in self._labelled_zip])

	def _fit(self, lb):
		# print self._n_possibilities
		n_dim = len(self._n_possibilities)
		n_category = len(self._cat_counter)
		p_category = self.get_prior_probability(lb)

		data = [None]*n_dim
		for dim, n_possibilities in enumerate(self._n_possibilities):
			data[dim] = [[(self._con_counter[dim][c][p]+lb)/(self._cat_counter[c]+lb*n_possibilities) for p in xrange(n_possibilities)] for c in xrange(n_category)]

		self._data = [np.array(dim_info) for dim_info in data]

		def func(input_x, tar_category):
			rs = 1
			for d, xx in enumerate(input_x):
				rs *= data[d][tar_category][xx]

			return rs*p_category[tar_category]

		return func

	def _transfer_x(self, x):
		for j, char in enumerate(x):
			x[j] = self._feat_dics[j][char]
		return x

def count_hashtag(tweets):
	hashtag_list = list()
	for tweet_str in tweets:
		tweet_list = tweet_str.lower().split()
		current_hashtag_list = [tweet_word[1:] for tweet_word in tweet_list if tweet_word[0] == '#']
		hashtag_list.extend(current_hashtag_list)
	hashtag_set = set(hashtag_list)
	print hashtag_set
	hashtag_dics = {_l:hashtag_list.count(_l) for _l in hashtag_set}
	return hashtag_dics

	# fig = plt.figure()
    # plt.title(title)
    # plt.barh(hashtag_dics.keys(),hashtag_dics.values(),width = 0.35,facecolor = 'lightskyblue',edgecolor = 'white')
    # plt.show()

if __name__ == '__main__':
	t = ['#a b c', '#a #b c','#a #b #c']
	print count_hashtag(t)
	import time
	from Util import DataUtil

	du = DataUtil()
	for dataset in ("balloon1.0(en)", "balloon1.5(en)"):
		_x, _y = du.get_dataset(dataset, "../_Data/{}.txt".format(dataset))
		learning_time = time.time()
		nb = MultinomialNB()
		nb.fit(_x, _y)
		learning_time = time.time() - learning_time

		estimation_time = time.time()
		nb.evaluate(_x,_y)
		estimation_time = time.time()-estimation_time

		print(
				"Model building : {:12.6}  s\n"
				"Estimation     : {:12.6}  s\n"
				"Total          : {:12.6}  s\n".format(learning_time, estimation_time, learning_time+estimation_time)
		)













