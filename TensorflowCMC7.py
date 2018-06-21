import tensorflow as tf
import numpy as np
import urllib
import cv2
import matplotlib.pyplot as plt
import random


# Hyperparameters
learning_rate = 0.001
num_steps = 100
batch_size = 128
display_step = 2
num_input = 4070 
num_classes = 15 
dropout = 0.75 

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):    
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 74, 55, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([28*19*32, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def print_all_images() : 
  for i in cmc : 
    plt.imshow(i)
    plt.show()
    
def print_image(img) : 
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  #plt.imshow(img)
  plt.show()
  

def url_to_image(url):	
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)   #cv2.IMREAD_COLOR
 
	return image

def load_cmc7() :
    
    cmc7List = []
    
    for i in range(0,15) :
      cmc7List.append(url_to_image("https://raw.githubusercontent.com/yoavalon/cmc7/master/" + str(i)+ ".png"))
    
    return np.array(cmc7List) 

#global datagenerator  
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=True,
    rotation_range=10.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=[0.4,1.2],
    channel_shift_range=200.,
    fill_mode='constant',
    cval=255.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)    
  
def random_augment(img) : 
  
  #print_image(img)    
  
  from keras import backend as k   
  k.set_image_dim_ordering('tf') 
  img = np.expand_dims(img, axis=2)  
  
  img = datagen.random_transform(img, seed=None)    
  img = img[:, :, 0]
    
  print_image(img)
  
  return img

#(128, 4070)
#(128, 15)
  
def create_Batch() : 
  
  imgList = []
  labels = []  
  
  for i in range(batch_size) :         
    ran = random.randint(0,14)  
    
    imgList.append(np.reshape(random_augment(cmc[ran]), 4070))
    labels.append(np.eye(15)[ran])  
  
  return np.array(imgList), np.array(labels)

cmc = load_cmc7()
print(cmc.shape)


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = create_Batch()        
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))



    testX, testY = create_Batch()        
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: testX,
                                      Y: testY,
                                      keep_prob: 1.0}))
    
