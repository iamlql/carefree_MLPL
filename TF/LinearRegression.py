import tensorflow as tf
import numpy as np
from Bases import ClassfierBase

class LinearRegression(ClassfierBase):
	def __init__(self):
		super(LinearRegression, self).__init__()
		self.w = tf.Variable(tf.zeros([2,1]), name = "weights")
		self.b = tf.Variable(0., name = "bias")

	def inference(self, x):
		return tf.matmul(x, self.w) + self.b

	def loss(self, x, y):
		y_pred = self.inference(x)
		return LinearRegression._l2_loss(y, y_pred)

	def inputs(self):
		weight_age = np.array([[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [76, 44]], dtype = np.float32)
		blood_fat_content = np.array([354, 190, 405, 263, 451, 302, 288, 385, 402, 365], dtype = np.float32)
		return weight_age, blood_fat_content[:,None]

	def evaluate(self, sess, x, y):
		print(sess.run(self.inference([[80., 25.]])))
		print(sess.run(self.inference([[65., 25.]])))

def main():
	lr = LinearRegression()
	lr.fit(learning_rate = 0.0000001)

if __name__ == "__main__":
	main()
