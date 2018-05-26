
import tensorflow as tf
import cv2



#     # init_op = tf.initialize_all_variables()
#     # sess.run(init_op)
    

# Create the graph, etc.
init_op = tf.initialize_all_variables()
tf_record_queue = tf.train.string_input_producer(tf.train.match_filenames_once('training-image.tfrecord'))
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_queue)

tf_record_features = tf.parse_single_example(tf_record_serialized, features = {'label': tf.FixedLenFeature([], tf.string), 'image': tf.FixedLenFeature([], tf.string)})
tf_record_image = tf.decode_raw(tf_record_features['image'], tf.uint8)
# tf_record_image = tf.reshape(tf_record_image, [height, width, channels])
tf_record_label = tf.cast(tf_record_features['label'], tf.string)

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run([tf_record_image, tf_record_label])

except tf.errors.OutOfRangeError:
    print ('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()