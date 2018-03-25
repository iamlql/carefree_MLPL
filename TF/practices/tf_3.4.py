import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():
	with tf.name_scope("Variables"):
		global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = "global_step")
		total_output = tf.Variable(0.0, dtype = tf.float32, trainable = False, name = "total_output")

	with tf.name_scope("Transformation"):
		with tf.name_scope("Input"):
			a = tf.placeholder(tf.float32,  name = "input_a")
		with tf.name_scope("intermediate_layer"):
			b = tf.reduce_prod(a, name = "prod")
			c = tf.reduce_sum(a, name = "sum")
		with tf.name_scope("output"):
			output = tf.add(b, c, name = "output")

	with tf.name_scope("Update"):
		update_total = total_output.assign_add(output)
		increment_step = global_step.assign_add(1)

	with tf.name_scope("Summaries"):
		avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name = "average")
		tf.summary.scalar("Output", output)
		tf.summary.scalar("Sum of outputs over time", update_total)
		tf.summary.scalar("Average of outputs over time", avg)

	with tf.name_scope("global_ops"):
		init = tf.global_variables_initializer()
		merged_summarys = tf.summary.merge_all()

	sess = tf.Session(graph = graph)
	writer = tf.summary.FileWriter('./my_graph3', graph = graph)
	sess.run(init)

	def run_graph(input_vector):
		feed_dict = {a: input_vector}
		_, step, summary = sess.run([output, increment_step, merged_summarys], feed_dict = feed_dict)
		print([_, step])
		writer.add_summary(summary, global_step = step)

	run_graph([2,8])
	run_graph([3,1,3,3])
	run_graph([8])
	run_graph([1,2,3])
	run_graph([11,3])

	writer.flush()

	sess.close()
	writer.close()
