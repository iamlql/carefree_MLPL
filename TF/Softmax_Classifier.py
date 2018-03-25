import tensorflow as tf
import numpy as np
from Bases import ClassfierBase
from Unity import DataUtil

class Softmax_Classifier(ClassfierBase):
	def __init__(self):
		super(Softmax_Classifier, self).__init__()
		self.w = tf.Variable(tf.zeros([4,3]), name = "weights")
		self.b = tf.Variable(tf.zeros([3]), name = "bias")

	def combine_inputs(self, x):
		return tf.matmul(x, self.w) + self.b

	def inference(self, x):
		return tf.nn.softmax(self.combine_inputs(x))

	def loss(self, x, y):
		y_pred = self.combine_inputs(x)
		return Softmax_Classifier._softmax_ce_loss(y_pred, y)

	def inputs(self):
		features, label = \
		DataUtil.get_dataset(100, "./data/iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
		label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([	\
			tf.equal(label, ["Iris-setosa"]), \
			tf.equal(label, ["Iris-versicolor"]), \
			tf.equal(label, ["Iris-virginica"])])), 0))
		return features, label_number

	def evaluate(self, sess, x, y):
		predicted = tf.cast(tf.argmax(self.inference(x), 1), tf.int32)
		print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))

def main():
	lr = Softmax_Classifier()
	lr.training_flow(learning_rate = 0.01, )

if __name__ == "__main__":
	main()