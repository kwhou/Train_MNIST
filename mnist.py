from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_one_hot_false = input_data.read_data_sets("MNIST_data/",one_hot=False)

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784,10], dtype=tf.float32))
b = tf.Variable(tf.random_normal([10], dtype=tf.float32))

y = tf.matmul(x,W)+b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_)
loss = tf.reduce_sum(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

w_boundary = 5
max_integer = 255
scale_up = max_integer / (2 * w_boundary)

clip_op = tf.assign(W, (tf.round((tf.clip_by_value(W, -w_boundary, w_boundary) + w_boundary) * scale_up) / scale_up) - w_boundary)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

sess.run(clip_op)

for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(500)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  sess.run(clip_op)

# print(sess.run(W).tolist())
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

x_data = mnist.test.images

accuracy_new = sess.run(accuracy, feed_dict={x: x_data, y_: mnist.test.labels})
print(accuracy_new)

h = open("pixels.txt",'w');
num_images = 100
h.write(str(num_images)+"\n")
for r in range(num_images):
  for k in range(784):
    h.write(str(int(mnist.test.images[r][k]*255))+"\n")
  h.write(str(mnist_one_hot_false.test.labels[r])+"\n")
h.close()


accuracy_prev = 0
g = open("accuracy.txt",'r')
accuracy_prev =float(g.readline())
g.close()

if (accuracy_new>accuracy_prev):
  g = open("accuracy.txt",'w')
  g.write(str(accuracy_new))
  g.close()
  
  print ("Writing new weights to file")
  
  weights = sess.run(W)
  
  f = open("weights.txt",'w')
  for i in range(0,784):
    for j in range(0,10):
      f.write(str( int((weights[i][j] + w_boundary) * scale_up))+" ")
    f.write("\n")
  f.close()
  