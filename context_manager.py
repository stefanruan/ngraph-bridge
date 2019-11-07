import tensorflow as tf
import numpy as np
from tensorflow.core.framework import attr_value_pb2
import ngraph_bridge


class ContextManager():

    def __init__(self, an_int):
        self.an_int = an_int

    def __enter__(self):
        graph = tf.get_default_graph()
        self.names = [i.name for i in graph.get_operations()]
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        graph = tf.get_default_graph()
        print(graph.get_operations())
        for node in graph.get_operations():
            if node.name not in self.names:
                node._set_attr('hello', attr_value_pb2.AttrValue(i=self.an_int))


# these nodes get tagged with "hello" attribute set to 5
with ContextManager(5) as manager:
    a = tf.constant(np.full((2, 2), 0.05, dtype=np.float32), name='alpha')
    x = tf.placeholder(tf.float32, [None, 2], name='x')
    y = tf.placeholder(tf.float32, shape=(2, 2), name='y')

# these nodes get tagged with "hello" attribute set to 6
with ContextManager(6) as manager:
    c = a * x
    axpy = c + y

for n in tf.get_default_graph().get_operations():
    print(n.name, n.get_attr('hello'))

with tf.Session() as sess:
    tf.io.write_graph(sess.graph, '.', 'OUT.pbtxt')
    sess.run(axpy, feed_dict={x: [[1, 2], [3, 4]], y: [[1, 2], [3, 4]]})

print('bye')
