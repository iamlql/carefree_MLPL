import tensorflow as tf
import numpy as np

a = tf.constant([5,3], name = 'input_a')
a2 =  np.array([2, 3], dtype = np.int32)
b = tf.reduce_prod(a, name = 'prod_b')
c = tf.reduce_sum(a, name = 'sum_c')
d = tf.add(a2, c, name = 'add_d')

sess = tf.Session()
print(sess.run([b, d]))

sess.close()