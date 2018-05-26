import tensorflow as tf
import numpy as np
from Bases import ClassfierBase
from Unity import DataUtil

class LinearSVM(ClassfierBase):
	def __init__(self):
		super(LinearSVM, self).__init__()
		self.w = tf.Variable(tf.zeros([5, 1]), name = "weights")
		self.b = tf.Variable(0., name = "bias")

	def combine_inputs(self, x):
		return tf.matmul(x, self.w) + self.b

	def inference(self, x):
		return tf.sign(self.combine_inputs(x))

	def loss(self, x, y):
		y_pred = self.combine_inputs(x)
		y_norm = tf.cast(2*y-1, tf.float32)
		return tf.reduce_sum(tf.maximum(1.0 - y_norm*y_pred, 0)) + tf.nn.l2_loss(self.w)

	def inputs(self):
		passenager_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
		DataUtil.read_csv(100, './data/train.csv', [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
		is_first_class = tf.to_float(tf.equal(pclass, [1]))
		is_second_class = tf.to_float(tf.equal(pclass, [2]))
		is_third_class = tf.to_float(tf.equal(pclass, [3]))

		gender = tf.to_float(tf.equal(sex, ["female"]))

		features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
		survived = tf.reshape(survived, [100,1])
		return features, survived

	def evaluate(self, sess, x, y):
		predicted = tf.cast(self.inference(x) > 0, tf.float32)
		print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))

def main():
	lr = LinearSVM()
	with tf.device("/gpu:0"):
		lr.fit(learning_rate = 0.0001, epoch = 5000)

if __name__ == "__main__":
	main()