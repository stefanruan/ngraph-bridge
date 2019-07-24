import tensorflow as tf
import numpy as np
import ngraph_bridge

def model1(dtype):
    x = tf.placeholder(dtype, [3, 3], name='x')
    y = tf.placeholder(dtype, [3, 3], name='y')
    a = tf.constant(np.full((3, 3), 1.5, dtype=np.float32), name='alpha', dtype=dtype)
    return tf.matmul(a, x) + y, [x, y]

def model2(dtype):
    x = tf.placeholder(dtype, [3, 3], name='x')
    y = tf.placeholder(dtype, [3, 3], name='y')
    a = tf.constant(np.full((3, 3), 1.5, dtype=np.float32), name='alpha', dtype=dtype)
    return (a * x) + y, [x, y]


def get_config(nativerun):
    config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)
    if nativerun:
        return config
    else:
        return ngraph_bridge.update_config(config)

models = [model1, model2]


for model in models:
    for dtype, nativerun in [(tf.float32, True), (tf.bfloat16, True), (tf.float32, False)]:
        out, list_of_ins = model(dtype)
        with tf.Session(config=get_config(nativerun)) as sess:
            if nativerun:
                ngraph_bridge.disable()
            else:
                ngraph_bridge.enable()
            # todo: replace np.ones with random values maybe
            feed_dict = {k: np.ones([i.value for i in k.shape.dims]) for k in list_of_ins}
            outval = sess.run(out, feed_dict=feed_dict)
            print(dtype, nativerun, outval[0][0])
            # TODO: get the diff of the matrices, instead of just printing out 1 number






