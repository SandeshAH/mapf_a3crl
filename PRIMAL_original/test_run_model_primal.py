# Test-only script
import tensorflow as tf
from ACNet import ACNet

sess = tf.Session()
net = ACNet("global", 5, None, False, 10, "global")
saver = tf.train.Saver()
saver.restore(sess, "model_primal/model-313300.cptk")
print("✅ Model loaded successfully!")
