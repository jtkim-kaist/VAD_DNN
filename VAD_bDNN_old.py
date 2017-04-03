import tensorflow as tf
import numpy as np
import utils as utils
import re
import datareader as dr
import os


FLAGS = tf.flags.FLAGS

file_dir = "/home/sbie/github/VAD_KJT/Data/data_0302_2017/Aurora2withSE"
input_dir = file_dir + "/Noisy_Aurora_STFT_npy"
output_dir = file_dir + "/label_checked"

valid_file_dir = "/home/sbie/github/VAD_KJT/Data/data_0308_2017/Aurora2withNX"
valid_input_dir = valid_file_dir + "/Noisy_Aurora_STFT_npy/Babble/SNR_10"
valid_output_dir = valid_file_dir + "/labels"

logs_dir = "/home/sbie/github/VAD_bDNN/logs"

reset = True  # remove all existed logs and initialize log directories
eval_only = False  # if True, skip the training phase
device = '/gpu:0'
if reset:
    os.popen('rm -rf ' + logs_dir + '/*')
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')

num_valid_batches = 100
learning_rate = 0.001
num_steps = 32
batch_size = 32  # batch_size = 32
num_sbs = 16  # number of sub-bands
fft_size = 256
channel = 1
cell_size = 256
cell_out_size = cell_size
num_h1_sb_net = 128
num_h2_sb_net = 256
SMALL_NUM = 1e-4
max_epoch = int(1e6)
dropout_rate = 0.85
decay = 0.9  # batch normalization decay factor
w = 19
u = 9
num_hidden_1 = 512
num_hidden_2 = 512

assert (w-1) % u == 0, "w-1 must be divisible by u"

num_features = 512
bdnn_winlen = (((w-1) / u) * 2) + 3
bdnn_inputsize = bdnn_winlen * num_features
bdnn_outputsize = bdnn_winlen


def affine_transform(x, output_dim, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """

    w = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name + "_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w) + b


def inference(inputs, keep_prob, is_training=True):

    # initialization

    h1_out = affine_transform(inputs, num_hidden_1, name="hidden_1")
    h1_out = tf.nn.relu(h1_out)
    h1_out = tf.nn.dropout(h1_out, keep_prob=keep_prob)

    h2_out = affine_transform(h1_out, num_hidden_2, name="hidden_2")
    h2_out = tf.nn.relu(h2_out)
    h2_out = tf.nn.dropout(h2_out, keep_prob=keep_prob)

    logits = affine_transform(h2_out, bdnn_outputsize, name="output")
    logits = tf.nn.sigmoid(logits)

    return logits


def train(loss_val, var_list):
    initLr = 5e-3
    lrDecayRate = .99
    lrDecayFreq = 200
    momentumValue = .9

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

    # define the optimizer
    # optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
    # optimizer = tf.train.AdagradOptimizer(lr)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)


def evaluation(m_valid, valid_data_set, sess, num_batches=100):
    # num_samples = valid_data_set.num_samples
    # num_batches = num_samples / batch_size
    avg_valid_cost = 0.
    avg_valid_reward = 0.
    for i in range(int(num_batches)):
        valid_inputs, valid_labels = valid_data_set.next_batch(batch_size)
        valid_inputs /= fft_size
        # print(train_labels.shape[0])
        valid_onehot_labels = dense_to_one_hot(valid_labels.reshape(-1, 1))
        valid_onehot_labels = valid_onehot_labels.reshape(-1, num_steps, 2)
        feed_dict = {m_valid.inputs: np.expand_dims(valid_inputs, axis=3), m_valid.raw_labels: valid_labels,
                     m_valid.onehot_labels: valid_onehot_labels, m_valid.keep_probability: 1}

        valid_cost, valid_reward = sess.run([m_valid.cost, m_valid.reward], feed_dict=feed_dict)

        avg_valid_cost += valid_cost
        avg_valid_reward += valid_reward

    avg_valid_cost /= (i + 1)
    avg_valid_reward /= (i + 1)

    return avg_valid_cost, avg_valid_reward


class Model(object):
    def __init__(self, is_training=True):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.inputs = inputs = tf.placeholder(tf.float32, shape=[batch_size, bdnn_inputsize],
                                              name="inputs")
        self.labels = labels = tf.placeholder(tf.float32, shape=[batch_size, bdnn_outputsize],
                                                            name="labels")

        # set inference graph
        logits = inference(inputs, self.keep_probability, is_training=is_training)  # (batch_size, bdnn_outputsize)
        # set objective function
        cost = tf.reduce_sum(tf.square(labels - logits), 1)
        self.cost = cost = tf.reduce_mean(cost)

        # set training strategy
        trainable_var = tf.trainable_variables()
        self.train_op = train(cost, trainable_var)


def main(argv=None):
    #                               Graph Part                               #
    print("Graph initialization...")
    with tf.device(device):
        with tf.variable_scope("model", reuse=None):
            m_train = Model(is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_valid = Model(is_training=False)
    print("Done")
    #                               Summary Part                             #
    with tf.variable_scope("summaries"):
        train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', max_queue=2)
        train_summary_list = [tf.summary.scalar("cost", m_train.cost)]
        train_summary_op = tf.summary.merge(train_summary_list)  # training summary

        avg_valid_cost = tf.placeholder(dtype=tf.float32)
        valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)
        valid_summary_list = [tf.summary.scalar("cost", avg_valid_cost)]
        valid_summary_op = tf.summary.merge(valid_summary_list)  # validation summary
    # Model Save Part                           #
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")
    #                               Session Part                              #
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:  # model restore
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    data_set = dr.DataReader(input_dir, output_dir, num_steps=num_steps,
                             name="train")  # training data reader initialization
    # data_set._num_file = 5
    valid_data_set = dr.DataReader(valid_input_dir, valid_output_dir,
                                   num_steps=num_steps, name="valid")  # validation data reader initialization

    for itr in range(max_epoch):
        train_inputs, train_labels = data_set.next_batch(batch_size)

        feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels,
                     m_train.keep_probability: dropout_rate}

        sess.run(m_train.train_op, feed_dict=feed_dict)

        if itr % 20 == 0:
            train_cost, train_reward, train_summary_str = sess.run([m_train.cost, m_train.reward, train_summary_op],
                                                                   feed_dict=feed_dict)
            print("Step: %d, train_cost: %.5f, train_reward: %.5f" % (itr, train_cost, train_reward))

            train_summary_writer.add_summary(train_summary_str, itr)  # write the train phase summary to event files

        if itr % 500 == 0:
            saver.save(sess, logs_dir + "/model.ckpt", itr)  # model save

            valid_cost, valid_reward = evaluation(m_valid, valid_data_set, sess, num_valid_batches)

            print("valid_cost: %.5f, valid_reward: %.5f" % (valid_cost, valid_reward))

            valid_summary_str = sess.run(valid_summary_op, feed_dict={avg_valid_cost: valid_cost,
                                                                      avg_valid_reward: valid_reward})
            valid_summary_writer.add_summary(valid_summary_str, itr)


if __name__ == "__main__":
    tf.app.run()


