import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import utils as ut 
tf.__version__

mnist_flat_size = 784


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data")
ut.plot_mnist_images(mnist.test.images[0:9])


tf.reset_default_graph()


z_dimention = 100

G_dimentions = [z_dimention,128,mnist_flat_size]

Z = tf.placeholder(tf.float32, shape=[None, G_dimentions[0]], name='x_generator_input')

def generator(x):
   
    with tf.name_scope("generator_nn"):
        g_net = tf.layers.dense(x,G_dimentions[1],
                                    activation=tf.nn.relu,
                                    name='g_hidden1',reuse=tf.AUTO_REUSE)
        for i in range(2,len(G_dimentions)-1):
            g_net = tf.layers.dense(g_net,G_dimentions[i],
                                    activation=tf.nn.relu,
                                    name='g_hidden'+str(i),reuse=tf.AUTO_REUSE)
        g_net = tf.layers.dense(g_net,G_dimentions[-1],activation=tf.nn.sigmoid,name='g_output',reuse=tf.AUTO_REUSE)
        return g_net


D_dimentions = [mnist_flat_size,128,1] 

X = tf.placeholder(tf.float32, shape=[None, D_dimentions[0]], name='x_discriminator_input')

def discriminator(x):
   
    with tf.name_scope("discriminator_nn"):
        d_net = tf.layers.dense(x,D_dimentions[1],
                                    activation=tf.nn.relu,
                                    name='d_hidden1',reuse=tf.AUTO_REUSE)
        for i in range(2,len(D_dimentions)-1):
            d_net = tf.layers.dense(d_net,D_dimentions[i],
                                    activation=tf.nn.relu,
                                    name='d_hidden'+str(i),reuse=tf.AUTO_REUSE)
        d_net = tf.layers.dense(d_net,D_dimentions[-1],activation=None,name='d_output',reuse=tf.AUTO_REUSE)
        
        d_logits = d_net
        d_net = tf.nn.sigmoid(d_net)
        return d_net,d_logits        

alternative_loss = True

with tf.name_scope("generator_loss"): 
    generated_sample = generator(Z)
    d_fake_prob, d_fake_logits = discriminator(generated_sample)
    
    if alternative_loss:
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
    else:
        g_loss = -tf.reduce_mean(tf.log(d_fake_prob))

with tf.name_scope("discriminator_loss"):
    d_true_prob, d_true_logits = discriminator(X)
    if alternative_loss:
        d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true_logits, labels=tf.ones_like(d_true_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
        d_loss = d_loss_fake + d_loss_true
    else:
        d_loss = -tf.reduce_mean(tf.log(d_true_prob) + tf.log(1. - d_fake_prob))



generator_variables = [var for var in tf.trainable_variables() if 'g_' in var.name]
discriminator_variables = [var for var in tf.trainable_variables() if 'd_' in var.name]


print(generator_variables)
print(discriminator_variables)

lr=0.001
with tf.name_scope("discriminator_train"):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    d_train_op = d_optimizer.minimize(d_loss, var_list=discriminator_variables) 

with tf.name_scope("generator_train"):
    g_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    g_train_op = g_optimizer.minimize(g_loss,var_list=generator_variables)


def random_Z(m, n=z_dimention):
    
    return np.random.uniform(-1., 1., size=[m, n])



batch_size = 1000
epoach = 10000

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoach):
    x_train,y_train = mnist.train.next_batch(batch_size)

   
    _, d_loss_value,d_sample = sess.run([d_train_op, d_loss, generated_sample], feed_dict={X: x_train, Z: random_Z(batch_size)})
  
    _, g_loss_value,g_sample = sess.run([g_train_op, g_loss,generated_sample], feed_dict={Z: random_Z(batch_size)})
    if i%100 == 0:
        print("Epoach",i)
        print("Discriminator loss: ",d_loss_value)
        print("Generator loss:",g_loss_value)
       
        z_new = random_Z(9)

        generated_images = generated_sample.eval(session=sess,feed_dict={Z: z_new})
        ut.plot_mnist_images(generated_images)


z_new = random_Z(9)

generated_images = generated_sample.eval(session=sess,feed_dict={Z: z_new})



