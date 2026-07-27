import tensorflow as tf

from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError

print("TensorFlow version:", tf.__version__)

# Test a contrib layer and LSTMCell
x = tf.placeholder(tf.float32, [None, 10])
cell = tf.contrib.rnn.BasicLSTMCell(32)
print("Successfully created contrib LSTMCell")

# with tf.Session() as sess:
#     print("Session started successfully")

# sess = tf.Session()
# print(sess.run(tf.test.is_gpu_available()))

print("GPU available:", tf.test.is_gpu_available())