import tensorflow as tf
import ngraph_bridge
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2

from tensorflow.python.ops import gen_array_ops as arr_ops

dim = 23948
x = tf.placeholder(tf.float32, shape=(dim))

qmin = 0.0
qmax = 10.0


x_q = arr_ops.quantize_and_dequantize_v2(x, qmin, qmax,
                                         signed_input=True, num_bits=8,
                                         range_given=True, name = 'input_quant')

#a = np.array([[40, 30, 20, 10], [10, 20, 15, 70],[23,25,70,10]], dtype=np.float32)
#y= tf.constant(a,dtype=tf.float32, shape=[3,4])
#a =  np.array([[0,1,2,4],[0,1,2,4]],dtype=np.float32)

a= np.random.uniform(qmin, qmax,dim)


b = tf.nn.top_k(x_q, 2, True)

#c = tf.slice(b.values, [0,0],[1,0])
#c = tf.cast(b.indices, tf.float32)
b_q =  arr_ops.quantize_and_dequantize_v2(b.values, qmin, qmax,
                                         signed_input=True, num_bits=8,
                                         range_given=True, name = 'input_quant')



config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1)
#config = ngraph_bridge.update_config(config,"INTERPRETER")
rewriter_options = rewriter_config_pb2.RewriterConfig()
rewriter_options.meta_optimizer_iterations=(rewriter_config_pb2.RewriterConfig.ONE)
rewriter_options.min_graph_nodes=-1
ngraph_optimizer = rewriter_options.custom_optimizers.add()

ngraph_optimizer.name = "ngraph-optimizer"
ngraph_optimizer.parameter_map["ngraph_backend"].s = b'CPU'
s="0"
ngraph_optimizer.parameter_map["device_id"].s = s.encode()

c = "'1'"
ngraph_optimizer.parameter_map["'num_ice_cores'"].s =c.encode() 
config.MergeFrom(tf.ConfigProto(graph_options=tf.GraphOptions(rewrite_options=rewriter_options)))

sess = tf.Session(config=config)
#print(sess.run([2*x], feed_dict={x:a}))
print(sess.run([b_q,b.indices],  feed_dict={x:a}))
#print(sess.run([b.values,b.indices],  feed_dict={x:a}))
#kth = tf.reduce_min(b.values)
#top2 = tf.greater_equal(a, kth)
#print(sess.run(top2))
sess.close()
