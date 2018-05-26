# Image loaded, and TF Record

import tensorflow as tf
import cv2

image_file = "dog_3184.jpg"
image_raw = tf.gfile.FastGFile(image_file,'rb').read()
image = tf.image.decode_jpeg(image_raw)
with tf.Session() as sess:
    # print(image.eval())
    image_label = b"\x01"
    image_loaded = image.eval()
    image_bytes = image_loaded.tobytes()
    height, width, channels = image_loaded.shape
    writer = tf.python_io.TFRecordWriter("training-image.tfrecord")
    example = tf.train.Example(features = tf.train.Features(feature={'label': tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_label])), 'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))}))
    writer.write(example.SerializeToString())
    writer.close()
    cv2.imshow("image", image.eval())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

