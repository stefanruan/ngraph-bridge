
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

from __future__ import print_function

import tensorflow as tf
import sys
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def get_forward_network():
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784], name='x') # mnist data image of shape 28*28=784

    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct forward model
    pred = tf.nn.softmax(tf.matmul(x, W) + b, name='pred') # Softmax
    return x, pred

def eval_model(pred, y):
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return accuracy

def save_sess_to_savedmodel(sess, location):
    builder = tf.saved_model.builder.SavedModelBuilder(location)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.TRAINING])
    builder.add_meta_graph([tf.saved_model.tag_constants.SERVING],
                            strip_default_attrs=True)
    builder.save()

def save_to_chkpt(sess, export_dir, step):
    saver = tf.train.Saver()
    saver.save(sess, export_dir, global_step=step)

def restore_from_chkpt(sess, import_dir):
    #saver = tf.train.Saver()
    #saver.restore(sess, import_dir) #"/tmp/model.ckpt"
    return tf.saved_model.load(sess, [tf.saved_model.tag_constants.TRAINING], import_dir)


# Parameters
learning_rate = 0.01
training_epochs = 3
batch_size = 100
display_step = 1
eval_step = 1

def generate_full_network_from_code():
    x, pred = get_forward_network()
    y = tf.placeholder(tf.float32, [None, 10], name='y') # 0-9 digits recognition => 10 classes
    accuracy = eval_model(pred, y)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1), name='cost')
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name='optimizer')
    return x, pred, y, accuracy, init, cost, optimizer

def get_tensors_from_graph(graph):
    return graph.get_tensor_by_name('x:0'), graph.get_tensor_by_name('pred:0'), graph.get_tensor_by_name('y:0'), graph.get_tensor_by_name('accuracy:0'), graph.get_tensor_by_name('cost:0'), graph.get_operation_by_name('optimizer')

def train_function(export_path, import_model):
    if export_path == 'mnist_model2':
        import ngraph_bridge

    # Start training
    with tf.Session() as sess:
        print("Num nodes in graph", len(sess.graph.get_operations()))
        if (import_model is not None):
            model = restore_from_chkpt(sess, import_model)
            x, pred, y, accuracy, cost, optimizer = get_tensors_from_graph(sess.graph)
        else:
            x, pred, y, accuracy, init, cost, optimizer = generate_full_network_from_code()
            # Run the initializer
            sess.run(init) 
            

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                            y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            if (epoch+1) % eval_step == 0:
                print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("Saving model")
        save_sess_to_savedmodel(sess, export_path)

        print("Optimization Finished!")
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        



if sys.argv[1] == '0':
    # Simulate training for the first time
    train_function("mnist_model", None)
elif sys.argv[1] == '1':
    # Simulate training from continuing from a half-trained mdoel
    train_function("mnist_model1", "mnist_model")
else:
    # load from tf2ngraph dumped graph
    train_function("mnist_model2", "../../OUT")



# https://www.tensorflow.org/guide/checkpoint
# checkpoints are now completely tied with keras