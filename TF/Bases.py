import tensorflow as tf
import numpy as np

class ClassfierBase:
	def __init__(self):
		self.name = self.__class__.__name__

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __getitem__(self, item):
		if isinstance(item, str):
			return getattr(self, "_"+item)

	def inputs(self):
		pass

	def inference(self, x):
		pass

	def loss(self, x, y):
		pass

	@staticmethod
	def _l2_loss(y, y_pred):
		return tf.reduce_sum(tf.squared_difference(y, y_pred))

	@staticmethod
	def _ce_loss(y_pred, y):
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits = y_pred, labels =y))

	@staticmethod
	def train(total_loss, learning_rate):
		return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

	def evaluate(self, sess, x, y):
		pass

	def training_flow(self, learning_rate = 0.001):
		
		with tf.name_scope("global_ops"):
			sess = tf.Session()
			init = tf.global_variables_initializer()
			sess.run(init)

		with tf.name_scope("Transformation"):
			with tf.name_scope("Input"):
				X, Y = self.inputs()

			with tf.name_scope("intermediate_layer"):
				total_loss = self.loss(X, Y)
				train_op = ClassfierBase.train(total_loss, learning_rate)
			with tf.name_scope("output"):
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess = sess, coord = coord)

		with tf.name_scope("Update"):
			training_steps = 1000
			# saver = tf.train.Saver()
			for step in np.arange(training_steps):
				sess.run([train_op])

				if step % 200 == 0:
					print("loss: {}".format(sess.run([total_loss])))
					# saver.save(sess, './backup/my_model', global_step = step)
			# saver.save(sess, './backup/my_model', global_step = training_steps)
			self.evaluate(sess, X, Y)

		with tf.name_scope("Summaries"):
			coord.request_stop()
			coord.join(threads)
			sess.close()