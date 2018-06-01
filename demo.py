import sys, json, base64
import numpy as np
import cv2
from io import BytesIO, StringIO

import tensorflow as tf
import vgg
import time

NUMBER_OF_CLASSES = 3
MODEL_PATH = "runs/curtrain/model.ckpt"
vgg_path = 'data/vgg'

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

def load_model(sess):
    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output, keep_prob, image_input = vgg.vgg16_fcn()

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy

    saver = tf.train.Saver()

    saver.restore(sess, MODEL_PATH)
    return model_output, keep_prob, image_input

def segment(logits, sess, image, keep_prob, image_pl):
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})

    result = np.argmax(im_softmax, axis = -1).reshape((image.shape[0], image.shape[1]))

    return ((result == 1).astype('uint8'), (result == 2).astype('uint8'))

# Define encoder function
def encode(array):
  retval, buff = cv2.imencode('.png', array)
  return base64.b64encode(buff).decode("utf-8")

video = cv2.VideoCapture(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

with tf.Session() as sess:
    logits, keep_prob, image_pl = load_model(sess)

    start = time.time()
    while (video.isOpened()):
      ret, bgr_frame = video.read()
      if ret:
        rgb_frame = bgr_frame[:, :, ::-1]
        # Look for red cars :)
        binary_car_result, binary_road_result = segment(logits, sess, rgb_frame, keep_prob, image_pl)

        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

        # Increment frame
        frame+=1
      else:
        break

    end = time.time()
    # print('Total time {} seconds to process {} frames'.format(end - start, frame))
    # Print output in proper json format
    print (json.dumps(answer_key))

video.release()
