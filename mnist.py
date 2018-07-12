from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_one_hot_false = input_data.read_data_sets("MNIST_data/",one_hot=False)

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.random_normal([10],dtype=tf.float32))

y = tf.matmul(x,W)+b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y ,labels = y_)
loss = tf.reduce_sum(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(2600):
  batch_xs, batch_ys = mnist.train.next_batch(500)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

x_data = mnist.test.images>0.0

accuracy_new = sess.run(accuracy, feed_dict={x:x_data, y_: mnist.test.labels})
print(accuracy_new)

weights = sess.run(W,feed_dict = {x:x_data,y_:mnist.test.labels})
h = open("pixels.txt",'w');

for r in range(10000):
  for k in range(784):
    h.write("\n")
    h.write(str(mnist.test.images[r][k]))
    h.write(str(mnist_one_hot_false.test.labels[r]))
h.close()

g = open("accuracy.txt",'r')
accuracy_prev =float(g.readline())
g.close()

if (accuracy_new>accuracy_prev):
  g = open("accuracy.txt",'w')
  g.write(str(accuracy_new))
  print ("Writing new weights to file")
  f = open("weights.txt",'w')
  f.write(str(weights[0][0]))
  for k in range(1,10): 
    f.write(" ")
    f.write(str(weights[0][k]))
  for i in range(1,784):
    f.write("\n")
    f.write(str(weights[i][0]))
    for j in range(1,10):
      f.write(" ")
      f.write(str(weights[i][j]))
  f.close()
  g.close()
     


