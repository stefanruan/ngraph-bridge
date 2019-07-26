import tensorflow as tf
import numpy as np
import ngraph_bridge
import os

# Tests the scenario below
# Reference: NativeTF + CPU
# - Input Tensor A,B,C,... : Dt.Float
# - Cast A,B,C,,,, to bfloat16
# - Run with TF
# - Output X
# This needs to be converted back to float32 for comparison
# Actual : TF with nGraph, NNP, Bo Funcsim
# - Same input tensors A,B,C : Dt.Float
# - Run with TF Bridge
# - Output Y
# - Do X and Y match with our reference tolerance (nnp.allclose/C++ Compare)


def model1(dtype):
    x = tf.placeholder(dtype, [3, 3], name='x')
    y = tf.placeholder(dtype, [3, 3], name='y')
    x = tf.cast(x, dtype=tf.bfloat16)
    y = tf.cast(y, dtype=tf.bfloat16)
    return tf.matmul(x, y), [x, y]


def model2(dtype):
    x = tf.placeholder(dtype, [3, 3], name='x')
    y = tf.placeholder(dtype, [3, 3], name='y')
    return tf.matmul(x, y), [x, y]


def get_config(nativerun):
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)
    if nativerun:
        return config
    else:
        return ngraph_bridge.update_config(config)


outputs = []
k_np = np.random.rand(3, 3)
for dtype, nativerun in [(tf.float32, True), (tf.float32, False)]:
    if nativerun:
        with tf.Session(config=get_config(nativerun)) as sess1:
            ngraph_bridge.disable()
            out, list_of_ins = model1(dtype)
            out = tf.cast(out, dtype=tf.float32)
            feed_dict = {k: k_np for k in list_of_ins}
            outval = sess1.run(out, feed_dict=feed_dict)
            outputs.append(outval[0][0])
    else:
        with tf.Session(config=get_config(nativerun)) as sess2:
            ngraph_bridge.enable()
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
            os.environ['NGRAPH_TF_BACKEND'] = 'NNP'
            out, list_of_ins = model2(dtype)
            feed_dict = {k: k_np for k in list_of_ins}
            outval = sess2.run(out, feed_dict=feed_dict)
            outputs.append(outval[0][0])
    print(out.dtype, nativerun, outval[0][0])

assert np.allclose(outputs[0], outputs[1])
