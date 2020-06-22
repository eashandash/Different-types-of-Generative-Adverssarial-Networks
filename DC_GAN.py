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
mnist = input_data.read_data_sets("./data",one_hot=True)

ut.plot_mnist_images(mnist.test.images[0:9])

z_dimention = 100
mnist_flat_size = 784

batch_size = 64
epoach = 50000

Z = tf.placeholder(tf.float32, shape=[None, z_dimention], name='x_generator_input')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='label_oneHot')
isTraining = tf.placeholder_with_default(False, shape=(), name='trainingPhase')

def generator(x,train,name):
   
    with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
        
        
        fc1 = tf.layers.dense(x, 7*7*64,name="fc1_"+name)
        fc1_bn = tf.layers.batch_normalization(fc1, training=train, name="fc1_bn_"+name)
        fc_relu1 = tf.nn.relu(fc1_bn,name="fc1_fn_"+name)
        
       
        r = tf.reshape(fc_relu1, [-1, 7, 7, 64])
        
        
        deconv1 = tf.layers.conv2d_transpose(r, 64, kernel_size=(5, 5), strides=(2, 2)
                                             ,padding='same',name="dconv1_"+name)
        deconv1_bn = tf.layers.batch_normalization(deconv1, training=train,name="dconv1_bn_"+name)
        dc_relu2 = tf.nn.relu(deconv1_bn,name="dconv1_fn_"+name)

        
        deconv2 = tf.layers.conv2d_transpose(dc_relu2, 1, kernel_size=(5, 5), strides=(2, 2),padding='same'
                                             ,name="dconv2_"+name, activation=tf.nn.tanh)

        return deconv2


X = tf.placeholder(tf.float32, shape=[None, mnist_flat_size], name='x_discriminator_input')

def discriminator(x,train,name,drop_rate=0.5):
   
    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
        
      
        r = tf.reshape(x, [-1, 28, 28, 1])
        
      
        conv1 = tf.layers.conv2d(r, 64, kernel_size=(5, 5), strides=(2, 2)
                                             ,padding='same',name="conv1_"+name)
        conv1_bn = tf.layers.batch_normalization(conv1, training=train, name="conv1_bn_"+name)
        c_lrelu1 = tf.nn.leaky_relu(conv1_bn,name="conv1_fn_"+name)
        
        
        conv2 = tf.layers.conv2d(c_lrelu1, 32, kernel_size=(5, 5), strides=(2, 2),padding='same'
                                             ,name="conv2_"+name)
        conv2_bn = tf.layers.batch_normalization(conv2, training=train, name="conv2_bn_"+name)
        c_lrelu2 = tf.nn.leaky_relu(conv2_bn, name="conv2_fn_"+name)
        
    
        flatten = tf.layers.flatten(c_lrelu2)
        
        fc1 = tf.layers.dense(flatten, 7*7*32,name="fc1_"+name,activation=tf.nn.leaky_relu)
        fc1_drop = tf.layers.dropout(fc1,rate = drop_rate, training=train)
        
        fc2 = tf.layers.dense(fc1_drop, 256,name="fc2_"+name,activation=tf.nn.leaky_relu)
        fc2_drop = tf.layers.dropout(fc2,rate = drop_rate, training=train)
        
        fc3 = tf.layers.dense(fc2_drop, 1,name="fc3_"+name, activation=None)
        
        sig_fc3 = tf.nn.sigmoid(fc3)
        
        return sig_fc3,fc3        



generated_sample = generator(Z,isTraining,"g_")
d_fake_prob, d_fake_logits = discriminator(generated_sample,isTraining,"d_")

X_tanh_normalize = (X-0.5)*2
d_true_prob, d_true_logits = discriminator(X_tanh_normalize,isTraining,"d_")

alternative_loss = True

print(d_true_logits)
print(d_fake_logits)
print(generated_sample)

eps = 1e-8

def xentropy_sigmoid(logits,labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

with tf.name_scope("discriminator_loss"):
    if alternative_loss:
        d_loss = xentropy_sigmoid(d_true_logits,tf.ones_like(d_true_logits)) + xentropy_sigmoid(d_fake_logits,tf.zeros_like(d_fake_logits)) 
    else:
        d_loss = -tf.reduce_mean(tf.log(d_true_prob + eps) + tf.log(1. - d_fake_prob + eps))


with tf.name_scope("generator_loss"):
    if alternative_loss:
        g_loss = xentropy_sigmoid(d_fake_logits, tf.ones_like(d_fake_logits))
    else:
        g_loss = -tf.reduce_mean(tf.log(d_fake_prob + eps))        



generator_variables = [var for var in tf.trainable_variables() if 'g_' in var.name]
discriminator_variables = [var for var in tf.trainable_variables() if 'd_' in var.name]


print(generator_variables)
print(discriminator_variables)

lr=0.001
with tf.name_scope("discriminator_train"):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.1)
    d_train_op = d_optimizer.minimize(d_loss, var_list=discriminator_variables) 
with tf.name_scope("generator_train"):
    g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.3)
    g_train_op = g_optimizer.minimize(g_loss,var_list=generator_variables)




sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoach+1):
    x_train,y_train = mnist.train.next_batch(batch_size)
    
  
    _, d_loss_value = sess.run([d_train_op, d_loss], feed_dict={X: x_train , Z: ut.random_Z(batch_size,n=z_dimention), isTraining:True})
  
    _, g_loss_value = sess.run([g_train_op, g_loss], feed_dict={Z: ut.random_Z(batch_size,n=z_dimention),isTraining:True})
    
    print("Epoach " + str(i), end="\r")
    
    if i%500 == 0:
        print("Discriminator loss: ",d_loss_value)
        print("Generator loss:",g_loss_value)
       
        z_new = ut.random_Z(9,n=z_dimention)
        print(z_new.shape)
        generated_images = generated_sample.eval(session=sess,feed_dict={Z: z_new}).reshape((9,784))

        ut.plot_mnist_images(generated_images)        