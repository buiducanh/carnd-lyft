#--------------------------
# USER-SPECIFIED DATA
#--------------------------

import tensorflow as tf
import time

DIVIDER = "============="

# Tune these parameters

NUMBER_OF_CLASSES = 3
IMAGE_SHAPE = (600, 800)
EPOCHS = 40
BATCH_SIZE = 10
DROPOUT = 0.75
SAVE_INTERVAL = BATCH_SIZE * 200

# Specify these directory paths

data_dir = './data'
runs_dir = './runs'
training_dir ='./lyft_training_data'
vgg_path = './data/vgg'

#--------------------------
# PLACEHOLDER TENSORS
#--------------------------

correct_label = tf.placeholder(tf.int32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

#--------------------------
# FUNCTIONS
#--------------------------

def load_vgg(sess, vgg_path):

  # load the model and weights
  model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

  # Get Tensors to be returned from graph
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    scaled3 = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
    scaled4 = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')

    weights_initializer_stddev = 0.01
    weights_regularized_l2 = 1e-3
    # Convolutional 1x1 to mantain space information.
    conv_1x1_of_7 = tf.layers.conv2d(vgg_layer7_out,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_7')
    # Upsample deconvolution x 2
    first_upsamplex2 = tf.layers.conv2d_transpose(conv_1x1_of_7,
                                                  num_classes,
                                                  4, # kernel_size
                                                  strides= (2, 2),
                                                  padding= 'same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                  name='first_upsamplex2')
    conv_1x1_of_4 = tf.layers.conv2d(scaled4,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_4')

    s1 = tf.shape(conv_1x1_of_4)
    s2 = tf.shape(first_upsamplex2)
    # Adding skip layer.
    first_skip = tf.add(
            first_upsamplex2[:, (s2[1] - s1[1]) // 2 : s1[1], (s2[2] - s1[2]) // 2 : s1[2], :],
            conv_1x1_of_4,
            name='first_skip'
    )
    # Upsample deconvolutions x 2.
    second_upsamplex2 = tf.layers.conv2d_transpose(first_skip,
                                                   num_classes,
                                                   4, # kernel_size
                                                   strides= (2, 2),
                                                   padding= 'same',
                                                   kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                   name='second_upsamplex2')
    conv_1x1_of_3 = tf.layers.conv2d(scaled3,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_3')

    s1 = tf.shape(conv_1x1_of_3)
    s2 = tf.shape(second_upsamplex2)
    # Adding skip layer.
    second_skip = tf.add(
            second_upsamplex2[:, (s2[1] - s1[1]) // 2 : s1[1], (s2[2] - s1[2]) // 2 : s1[2], :],
            conv_1x1_of_3,
            name='second_skip'
    )
    # Upsample deconvolution x 8.
    third_upsamplex8 = tf.layers.conv2d_transpose(second_skip, num_classes, 16,
                                                  strides= (8, 8),
                                                  padding= 'same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                  name='third_upsamplex8')

    return third_upsamplex8

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

  # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
  correct_label_reshaped = tf.reshape(correct_label, [-1])

  # Calculate distance from actual labels using cross entropy
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped)
  # Take mean for total loss
  loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

  # The model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

  return logits, train_op, loss_op

def save_model(sess, saver, start, save_ind):
  model_name = os.path.join(runs_dir, "model_{}.ckpt".format(save_ind))
  save_path = saver.save(sess, model_name)
  print(DIVIDER)
  print("Saving model to {}".format(save_path))
  print("TIME elapsed {}...".format(time.time() - start))

def train_nn(saver, sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

  learning_rate_value = 0.001

  start = time.time()

  img_cnt = 0
  save_ind = 0
  for epoch in range(epochs):
      start_epoch = time.time()

      # Create function to get batches
      total_loss = 0
      for X_batch, gt_batch in get_batches_fn(batch_size):
          loss, _ = sess.run([cross_entropy_loss, train_op],
          feed_dict={input_image: X_batch, correct_label: gt_batch,
          keep_prob: DROPOUT, learning_rate:learning_rate_value})
          total_loss += loss

          img_cnt = (img_cnt + batch_size) % SAVE_INTERVAL

          if img_cnt == 0:
              save_ind += 1
              save_model(sess, saver, start, save_ind)

      end_epoch = time.time()

      print(DIVIDER)
      print("EPOCH {} ...".format(epoch + 1))
      print("Training time: {}".format(end_epoch - start_epoch))
      print("Loss = {:.3f}".format(total_loss))
      print()

  end = time.time()
  print(DIVIDER)
  print("Total training time:  {}".format(end - start))


import helper

def test():
    with tf.Session() as sess:
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)

        saver = tf.train.Saver()

        saver.restore(sess, runs_dir + '/model.ckpt')

        helper.save_inference_samples(runs_dir, training_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)

        print("All done!")

def run():

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # A function to get batches
  get_batches_fn = helper.gen_batch_function(training_dir, IMAGE_SHAPE)

  with tf.Session() as session:

    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)

    saver = tf.train.Saver()

    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    print("Model build successful, starting training")

    # Train the neural network
    train_nn(saver, session, EPOCHS, BATCH_SIZE, get_batches_fn,
             train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

    save_path = saver.save(session, runs_dir + '/model.ckpt')
    print('Model save at {}'.format(save_path))

    # Run the model with the test images and save each painted output image (roads painted green)
    helper.save_inference_samples(runs_dir, training_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)

    print("All done!")

def data():
    helper.maybe_download_lyft_data(training_dir)

#--------------------------
# MAIN
#--------------------------
if __name__ == '__main__':
    # test()
    run()
    # data()
