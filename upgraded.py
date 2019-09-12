import pandas as pd
import tensorflow as tf
import numpy as np

import tensorflow.python.framework.dtypes

a = tf.constant(2)
b = tf.constant(3)
d = a*b

with tf.compat.v1.Session() as sess:
    sess.run(d)
