# This program aims to generate fake noisy images.



# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from os import listdir
from os.path import isfile, join

from PIL import Image

times_repeat = 100 # number of epochs

batch_size = 1 # batch size
sizeimg = 28
X_dim = sizeimg*sizeimg
z_dim = 64 # dimensions of noise
h_dim = 128
lr = 1e-4
# d_steps = 3
# noise_sigma = 25

name_net = 'softmax_gan_X_Gcnn_Dfc_evoluted_report07_using'
summaries_dir = "/home/fun/softmax_gan/summaries/" + name_net
model_dir = "/home/fun/softmax_gan/summaries/" + name_net + '_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# In[2]:


### Load data
path_data_clean = '/home/tai-databases/mnist/mnist_png/'
path_data_noisy = '/home/tai-databases/mnist/mnist_png_gaus25/'
path_data_noisy_fake = '/home/tai-databases/mnist/mnist_png_gaus25_fake/'
if not os.path.exists(path_data_noisy_fake):
    os.makedirs(path_data_noisy_fake)

path_train_listname = '/home/tai-databases/mnist/mnist_png_gaus25paired/train/'
path_val_listname = '/home/tai-databases/mnist/mnist_png_gaus25paired/val/'
path_test_listname = '/home/tai-databases/mnist/mnist_png_gaus25paired/test/'

list_img_name_full = [f for f in listdir(path_data_clean) if isfile(join(path_data_clean, f))]
list_img_name_train = [f for f in listdir(path_train_listname) if isfile(join(path_train_listname, f))]
list_img_name_val = [f for f in listdir(path_val_listname) if isfile(join(path_val_listname, f))]
list_img_name_test = [f for f in listdir(path_test_listname) if isfile(join(path_test_listname, f))]

# num_of_images = len(list_img_name_train)
print('training images:', len(list_img_name_train))
print('validating images:', len(list_img_name_val))
print('testing images:', len(list_img_name_test))


def NormalizeData(dataset):
#     print('origin max:', np.amax(dataset))
#     dataset = dataset - (dataset.mean())
#     dataset = dataset / (dataset.std())
#     print('after Normalized:', np.amax(dataset))
    max_value = np.amax(dataset).astype(float)
    dataset = dataset/max_value
#     print('after divided:', np.amax(dataset))    
    return dataset

### listname to images tensor
def LoadTensorFromImages(path,listname, start, stop):
    if stop > len(listname):
        stop = len(listname)
    tensor  = np.array([np.array(Image.open(path+fname)) for fname in listname[start:stop]])
    tensor_reshape = np.reshape(tensor,(tensor.shape[0], tensor.shape[1], tensor.shape[2],1))
    return NormalizeData(tensor_reshape)


def saveImgWithPIL(img, path):
    ### input is an array range 0.0 to 1.1
    img *= 255.0/img.max()
    img = np.uint8(img)
    img_pil = Image.fromarray(img)
    img_pil.save(path)


# In[3]:


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
    return tf.log(x + 1e-8)


def sample_z(shape):
    return np.random.uniform(-1., 1., size=shape)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# def G(z):
#     with tf.name_scope('G'):
#         G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#         G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#         G_prob = tf.nn.sigmoid(G_log_prob)
#         return G_prob

def G(z):
    with tf.name_scope('G'):
        G_conv1 = tf.nn.relu(conv2d(z, G_W1) + G_b1)
        G_conv2 = tf.nn.relu(conv2d(G_conv1, G_W2) + G_b2)
        G_conv3 = tf.nn.relu(conv2d(G_conv2, G_W3) + G_b3)
        G_conv4 = conv2d(G_conv3, G_W4) + G_b4
#         print(tf.shape(G_conv4))
        return G_conv4


def D(X):
    with tf.name_scope('D'):
        G_conv1 = max_pool_2x2(tf.nn.relu(conv2d(X, D_cW1) + D_cb1))
        X_flat = tf.reshape(G_conv1, [-1,14*14*16])
        D_h1 = tf.nn.relu(tf.matmul(X_flat, D_W1) + D_b1)
        out = tf.matmul(D_h1, D_W2) + D_b2
        return out

def add_gauss_noise(image, sigma):
#     row,col,ch= image.shape
#     image_shape = image.shape
    mean = 0
    sigma = sigma/255.0
    gauss = np.random.normal(mean,sigma,(image.shape))
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    return noisy    

def add_noise_multi_images(images, sigma):
    images_noisy = np.zeros_like(images)
    for i in range(len(images)):
        images_noisy[i] = add_gauss_noise(images[i], sigma)
    return images_noisy
    
graphCNN = tf.Graph()
with graphCNN.as_default():    

    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    img_noisy_real = tf.summary.image('img_noisy_real', tf.reshape(X, shape=[-1, 28, 28, 1]), 3)
    
    X_ =  tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    img_clear = tf.summary.image('img_clear', tf.reshape(X_, shape=[-1, 28, 28, 1]), 3)
    
    z = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='z')

#     ### Generator's Parameters
#     G_W1 = tf.Variable(xavier_init([z_dim, h_dim]), name='G_W')
#     G_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name='G_b')
#     G_W2 = tf.Variable(xavier_init([h_dim, X_dim]), name='G_W')
#     G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name='G_b')

    ### Generator's Parameters
    G_W1 = weight_variable([5, 5, 1, 16], name='G_W1')
    G_b1 = bias_variable([16], name='G_b1')
    G_W2 = weight_variable([5, 5, 16, 64], name='G_W2')
    G_b2 = bias_variable([64], name='G_b2')
    G_W3 = weight_variable([5, 5, 64, 16], name='G_W3')
    G_b3 = bias_variable([16], name='G_b3')
    G_W4 = weight_variable([5, 5, 16, 1], name='G_W4')
    G_b4 = bias_variable([1], name='G_b4')
    
    G_W1_hist = tf.summary.histogram('G_W1', G_W1)
    G_W2_hist = tf.summary.histogram('G_W2', G_W2)
    G_W3_hist = tf.summary.histogram('G_W3', G_W3)
    G_W4_hist = tf.summary.histogram('G_W4', G_W4)
    G_b1_hist = tf.summary.histogram('G_b1', G_b1)
    G_b2_hist = tf.summary.histogram('G_b1', G_b2)
    G_b3_hist = tf.summary.histogram('G_b3', G_b3)
    G_b4_hist = tf.summary.histogram('G_b4', G_b4)
    
    ### Discriminator's Parameters
    D_cW1 = weight_variable([5, 5, 1, 16], name='G_cW1')
    D_cb1 = bias_variable([16], name='G_cb1')
    D_W1 = weight_variable([14*14*16, h_dim], name='D_W1')
    D_b1 = bias_variable(shape=[h_dim],name='D_b1')
    D_W2 = weight_variable([h_dim, 1],name='D_W2')
    D_b2 = bias_variable(shape=[1], name='D_b2')

    D_cW1_hist = tf.summary.histogram('D_cW1', D_cW1)
    D_cb1_hist = tf.summary.histogram('D_cb1', D_cb1)
    D_W1_hist = tf.summary.histogram('D_W1', D_W1)
    D_b1_hist = tf.summary.histogram('D_b1', D_b1)
    D_W2_hist = tf.summary.histogram('D_W2', D_W2)
    D_b2_hist = tf.summary.histogram('D_b2', D_b2)
    
    theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_W4, G_b4]
    theta_D = [D_cW1, D_cb1, D_W1, D_b1, D_W2, D_b2]
    
#     img_noisy_fake = tf.add(G(z),X)
    Gz = G(z)
#     print(Gz.get_shape)
#     print(X_.get_shape)
    G_sample2 = tf.clip_by_value(tf.add(Gz,X_),0.0,1.0)
    
    img_noisy_fake = tf.summary.image('img_noisy_fake', tf.reshape(G_sample2, shape=[-1, 28, 28, 1]), 3)
    noise_generated = tf.summary.image('noise_generated', tf.reshape(Gz, shape=[-1, 28, 28, 1]), 3)
    
    D_real = D(X)
    D_fake = D(G_sample2)
        
    D_target = 1./batch_size
    G_target = 1./(batch_size*2)

    Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))

    D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
    D_loss_summary = tf.summary.scalar("Loss_D", D_loss)
    
    G_loss = tf.reduce_sum(G_target * D_real) + tf.reduce_sum(G_target * D_fake) + log(Z)
    G_loss_summary = tf.summary.scalar("Loss_G", G_loss)
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G))

#     merged_sumarry = tf.summary.merge([img_noisy_fake, img_noisy_real, img_clear, noise_generated])
    merged_sumarry = tf.summary.merge_all()
    image_summarry = tf.summary.merge([img_clear, img_noisy_fake, noise_generated] )
        
    sess = tf.InteractiveSession(graph=graphCNN)
    sess.run(tf.global_variables_initializer())
    
    # Create a saver.
    saver = tf.train.Saver(max_to_keep=500)
    saver.restore(
        sess, 
        "/home/fun/softmax_gan/summaries/softmax_gan_X_Gcnn_Dfc_evoluted_report07_remake_model/model-100000"
    )

    merge_writer = tf.summary.FileWriter(summaries_dir + '/merged')
    merge_writer.add_graph(sess.graph)

    global_step = 0

    for step in range(len(list_img_name_full)):
        global_step+=1

        z_mb = sample_z([batch_size, 28, 28, 1]) # make a sample noise
        X_mb = LoadTensorFromImages(path_data_clean, list_img_name_full, step, step+batch_size)    
#             X_mb_noised = LoadTensorFromImages(path_data_noisy, list_img_name_train, step, step+batch_size)

#             _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={z: z_mb, X_: X_mb})
#             _, G_loss_curr = sess.run([G_solver, G_loss] , feed_dict={z: z_mb, X_: X_mb})

        X_mb_noised = sess.run(G_sample2, feed_dict={z: z_mb, X_: X_mb})
    
        X_mb_noised_tosave = np.reshape(X_mb_noised, [sizeimg,sizeimg])
        
        saveImgWithPIL(X_mb_noised_tosave, path_data_noisy_fake + list_img_name_full[step])

        if global_step % 50 == 0:
#                 print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(global_step, D_loss_curr, G_loss_curr))
            s  = sess.run(image_summarry, feed_dict={z: z_mb, X_: X_mb})
            merge_writer.add_summary(s, global_step)

#                 if global_step % 100000 == 0:
#                     # Append the step number to the checkpoint name:
#                     saver.save(sess, model_dir + '/model', global_step=global_step)


# In[4]:


print(np.reshape(X_mb_noised, [28,28]).shape)

