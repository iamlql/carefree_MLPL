import tensorflow as tf
# import cv2

import glob
from itertools import groupby
from collections import defaultdict


image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

image_filename_with_breed = map(lambda filename:(filename.split("/")[2], filename), image_filenames)

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
	for i, breed_images in enumerate(breed_images):
		if i % 5 == 0:
			testing_dataset[dog_breed].append(breed_images[1])
		else:
			training_dataset[dog_breed].append(breed_images[1])

	breed_training_count = len(training_dataset[dog_breed])
	breed_testing_count = len(testing_dataset[dog_breed])

	assert round(bread_testing_count / (bread_testing_count + bread_training_count), 2) > 0.18, "No enough testing images."



# sess = tf.Session()
# image_filename = 'dog_3184.jpg'
# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))
# image_reader = tf.WholeFileReader()
# _, image_file = image_reader.read(filename_queue)
# image_batch = tf.image.decode_jpeg(image_file)


# print(sess.run(image_batch))

# kernel = tf.constant([
# [
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
# ],
# [
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
# 	[[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
# 	],
# [
# [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
# 	[[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
# ]
# 	])

# conv2d = tf.nn.conv2d(image_batch, kernel, [1, 1, 1, 1], padding = "SAME")

# activate_map = sess.run(tf.minimum(tf.nn.relu(conv2d)), 255)