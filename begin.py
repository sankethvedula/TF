import tensorflow as tf


hello = tf.constant("Hello TensorFlow!")


sess = tf.Session()

print sess.run(hello)

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
	print "a=2, b=3"
	print "Adition = %i" %sess.run(a+b)
	print "multiplication = %i" %sess.run(a*b)


add = tf.add(a,b)
mul = tf.mul(a,b)

with tf.Session() as sess:
        print "a=2, b=3"
        print "Adition = %i" %sess.run(add, feed_dict = {a:2, b:3})
        print "multiplication = %i" %sess.run(mul, feed_dict = {a:2, b:3})



matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)


with tf.Session() as sess:
	result = sess.run(product)
	print result
