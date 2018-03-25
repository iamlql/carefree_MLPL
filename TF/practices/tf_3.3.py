import tensorflow as tf
import numpy as np

# with tf.name_scope("Scope_A"):
# 	a = tf.placeholder(tf.int32, shape=[2], name = "input_a")
# 	b = tf.multiply(a, 3, name = "mul_b")

# with tf.name_scope("Scope_B"):
# 	c = tf.add(4, 5, name = "add_c")
# 	d = tf.div(c, 6, name = "div_d")

# e = tf.add(b, d, name = "output_e")

# sess = tf.Session()
# print(sess.run(e, {a: np.array([1, 2])}))

# writer = tf.summary.FileWriter('./my_graph1', graph = tf.get_default_graph())
# writer.close()

graph = tf.Graph()

with graph.as_default():
	in_1 = tf.placeholder(tf.float32, shape = [2], name = 'input_a')
	in_2 = tf.placeholder(tf.float32, shape = [2], name = 'input_b')

	const = tf.constant(3, dtype = tf.float32, name = "static_value")

	with tf.name_scope("Transformation"):
		with tf.name_scope("A"):
			A_mul = tf.multiply(in_1, const)
			A_out = tf.subtract(A_mul, in_1)

		with tf.name_scope("B"):
			B_mul = tf.multiply(in_2, const)
			B_out = tf.subtract(B_mul, in_2)

		with tf.name_scope("C"):
			C_div = tf.div(A_out, B_out)
			C_out = tf.add(C_div, const)

		with tf.name_scope("D"):
			D_div = tf.div(B_out, A_out)
			D_out = tf.add(D_div, const)

	out = tf.maximum(C_out, D_out, name = "maximum")

	sess = tf.Session()
	print(sess.run(out, {in_1: np.array([1, 2]), in_2: np.array([3,4])}))


	writer = tf.summary.FileWriter('./my_graph2', graph = graph)
	writer.close()

