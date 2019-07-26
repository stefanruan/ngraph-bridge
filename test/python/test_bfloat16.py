import tensorflow as tf
import numpy as np
import ngraph_bridge
import os

# Tests the scenario below
# Reference: NativeTF + CPU
# - Input Tensor X,Y ... : Dt.Float
# - Cast X,Y to bfloat16
# - MatMul takes in X,Y->outputs A (bfloat16)
# - Cast A to Dt.Float

# Actual : TF with nGraph, NNP, B0 Funcsim
# - Same input tensors X,Y : Dt.Float
# -  MatMul takes in X,Y ->outputs B (Dt.Float)

# Compare A and B with tolerance


def tf_model():
    x = tf.placeholder(tf.float32, [3, 3], name='x')
    y = tf.placeholder(tf.float32, [3, 3], name='y')
    x = tf.cast(x, dtype=tf.bfloat16)
    y = tf.cast(y, dtype=tf.bfloat16)
    m = tf.matmul(x, y)
    m = tf.cast(m, dtype=tf.float32)
    return m, [x, y]


def ng_model():
    x = tf.placeholder(tf.float32, [3, 3], name='x')
    y = tf.placeholder(tf.float32, [3, 3], name='y')
    m = tf.matmul(x, y)
    return m, [x, y]


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

k_np = np.random.rand(3, 3)

#Test 1: tf_model TF-native
with tf.Session(config=config) as sess_tf:
    ngraph_bridge.disable()
    tf_out, list_of_ins = tf_model()
    feed_dict = {k: k_np for k in list_of_ins}
    tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)
    print("Native TF: ")
    print(tf_outval.dtype, tf_outval[0][0])

#Test 2: model2 with ngraph, NNP backend
with tf.Session(config=config) as sess_ng:
    ngraph_bridge.enable()
    ngraph_bridge.update_config(config)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
    os.environ['NGRAPH_TF_BACKEND'] = 'NNP'
    ng_out, list_of_ins = ng_model()
    feed_dict = {k: k_np for k in list_of_ins}
    ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)
    print("Ngraph with NNP backend: ")
    print(ng_outval.dtype, ng_outval[0][0])

try:
    assert np.allclose(tf_outval[0][0], ng_outval[0][0])
    print(" \033[92m PASS \033[0m ")
except:
    print(" \033[91m FAIL \033[0m ")
