import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
n_values = 32

x = tf.linspace(-3.0,3.0,n_values)

sess = tf.Session()

result = sess.run(x)

x.eval(session=sess)

# Interactive Session where you don't have to pass the session to the eval function

sess.close()
sess = tf.InteractiveSession()
x.eval()


# Now tf.Operation 
# Use the values from [-3.,3.] to create a Gaussian Distribution

sigma = 1.0
mean = 0.0

z = (tf.exp(tf.neg(tf.pow(x-mean,2.0)/(2.0*tf.pow(sigma,2.0))))* (1.0/(sigma*tf.sqrt(2.0*3.1415))))

#new operations are added to the tf.get_default_graph()

assert z.graph is tf.get_default_graph()

#plt.plot(z.eval())

print(z.get_shape())
print(z.get_shape().as_list())


# When we do not know the shape of the tensor we can use tf.shape(--tensor--).eval()

print(tf.shape(z).eval())
print(tf.pack([tf.shape(z), tf.shape(z), [3],[4]]).eval())

z_2d = tf.matmul(tf.reshape(z,[n_values,1]),tf.reshape(z, [1,n_values]))

plt.plot(z_2d.eval())

plt.savefig('saved.png')
