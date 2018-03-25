import tensorflow as tf
import numpy as np


g1 = tf.Graph()
# g2 = tf.Graph()

with g1.as_default():
	a = tf.placeholder(tf.int32, shape=[2], name = "input_a")
	b = tf.constant(3, name = "input_b")
	var1 = tf.Variable(0, name = 'v1')
	var2 = tf.Variable(1, name = 'v2')
	d = tf.add(a, b, name = "add_d")
	c = tf.multiply(a, b, name = "mul_c")
	e = tf.add(c, d, name = "add_e")
	f = tf.add(e, var1, name = "add_f")

	init = tf.initialize_all_variables()


	sess = tf.Session()
	sess.run(init)
	print(sess.run(e, {a: np.array([1, 2])}))
	sess.run(var1.assign_add(3))
	print(sess.run(f, {a: np.array([3, 2])}))




writer = tf.summary.FileWriter('./my_graph', g1)
writer.close()
sess.close()