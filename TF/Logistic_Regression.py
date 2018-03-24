import tensorflow as tf
import numpy as np
import os
from Bases import ClassfierBase

class Logistic_Regression(ClassfierBase):
	def __init__(self):
		super(Logistic_Regression, self).__init__()
		self.w = tf.Variable(tf.zeros([5, 1]), name = "weights")
		self.b = tf.Variable(0., name = "bias")

	def combine_inputs(self, x):
		return tf.matmul(x, self.w) + self.b

	def inference(self, x):
		return tf.sigmoid(self.combine_inputs(x))

	def loss(self, x, y):
		y_pred = self.combine_inputs(x)
		return Logistic_Regression._ce_loss(y_pred, y)

	@staticmethod
	def read_csv(batch_size, file_name, record_defaults):
		filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)+'/'+file_name])
		reader = tf.TextLineReader(skip_header_lines = 1)
		key, value = reader.read(filename_queue)

		decoded = tf.decode_csv(value, record_defaults = record_defaults)
		return tf.train.shuffle_batch(decoded, batch_size = batch_size, capacity = batch_size*50, min_after_dequeue = batch_size)

	def inputs(self):
		passenager_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
		Logistic_Regression().read_csv(100, './data/train.csv', [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
		is_first_class = tf.to_float(tf.equal(pclass, [1]))
		is_second_class = tf.to_float(tf.equal(pclass, [2]))
		is_third_class = tf.to_float(tf.equal(pclass, [3]))

		gender = tf.to_float(tf.equal(sex, ["female"]))

		features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))

		survived = tf.reshape(survived, [100,1])
		return features, survived

	def evaluate(self, sess, x, y):
		predicted = tf.cast(self.inference(x) > 0.5, tf.float32)
		print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))

def main():
	lr = Logistic_Regression()
	lr.training_flow(learning_rate = 0.01)

if __name__ == "__main__":
	main()

