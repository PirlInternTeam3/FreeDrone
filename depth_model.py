# -*- coding: utf-8 -*-

from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import os
import psutil
import time

cwd = os.getcwd()
path=cwd+'/images/00001_org.png'
print(path)

process = psutil.Process(os.getpid())

sess=tf.Session()
sess1=tf.Session()

#First let's load meta graph and restore weights
tf.global_variables_initializer()
saver = tf.train.import_meta_graph('./refine/model.ckpt-99.meta')
saver.restore(sess,'./refine/model.ckpt-99')
saver1 = tf.train.import_meta_graph('./coarse/model.ckpt-99.meta')
saver1.restore(sess1,'./coarse/model.ckpt-99')
val1=sess1.run('coarse1/biases:0')
val2=sess1.run('coarse1/weights:0')

val3=sess1.run('coarse2/biases:0')
val4=sess1.run('coarse2/weights:0')

val5=sess1.run('coarse3/biases:0')
val6=sess1.run('coarse3/weights:0')

val7=sess1.run('coarse4/biases:0')
val8=sess1.run('coarse4/weights:0')

val9=sess1.run('coarse5/biases:0')
val10=sess1.run('coarse5/weights:0')

val11=sess1.run('coarse6/biases:0')
val12=sess1.run('coarse6/weights:0')

val13=sess1.run('coarse7/biases:0')
val14=sess1.run('coarse7/weights:0')


fval1=sess.run('fine1/biases:0')
fval2=sess.run('fine1/weights:0')

fval3=sess.run('fine3/biases:0')
fval4=sess.run('fine3/weights:0')

fval5=sess.run('fine4/biases:0')
fval6=sess.run('fine4/weights:0')
input_tensor = tf.placeholder(tf.float32, [1,228,304,3])      
conv1 = tf.nn.conv2d(input_tensor,val2, [1, 4, 4, 1], padding='VALID')
bias1 = tf.nn.bias_add(conv1, val1)
coarse1_conv= tf.nn.relu(bias1)
coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

conv2=tf.nn.conv2d(coarse1,val4,[1, 1, 1, 1],padding='VALID')
bias2=tf.nn.bias_add(conv2,val3)
coarse2_conv=tf.nn.relu(bias2)
coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


conv3=tf.nn.conv2d(coarse2,val6,[1, 1, 1, 1],padding='VALID')
bias3=tf.nn.bias_add(conv3,val5)
coarse3=tf.nn.relu(bias3)

conv4=tf.nn.conv2d(coarse3,val8,[1, 1, 1, 1],padding='VALID')
bias4=tf.nn.bias_add(conv4,val7)
coarse4=tf.nn.relu(bias4)

conv5=tf.nn.conv2d(coarse4,val10,[1, 1, 1, 1],padding='VALID')
bias5=tf.nn.bias_add(conv5,val9)
coarse5=tf.nn.relu(bias5)


shape1=6*10*256
flat1 = tf.reshape(coarse5, [-1,shape1])
coarse6=tf.nn.relu_layer(flat1,val12,val11)

shape2=4096
flat2 = tf.reshape(coarse6, [-1,shape2])
coarse7=tf.nn.relu_layer(flat2,val14,val13)

coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])

fconv1=tf.nn.conv2d(input_tensor,fval2,[1, 2, 2, 1],padding='VALID')
fbias1 = tf.nn.bias_add(fconv1,fval1)
fine1_conv=tf.nn.relu(fbias1)
fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='fine_pool1')

# keep_conv=1.0
# fine1_dropout = tf.nn.dropout(fine1, keep_conv)
fine2 = tf.concat([fine1, coarse7_output],3)

fconv3=tf.nn.conv2d(fine2,fval4,[1, 1, 1, 1],padding='SAME')
fbias3 = tf.nn.bias_add(fconv3,fval3)
fine3=tf.nn.relu(fbias3)
# fine3_dropout = tf.nn.dropout(fine3, keep_conv)

fconv4=tf.nn.conv2d(fine3,fval6,[1, 1, 1, 1],padding='SAME')
fbias4=tf.nn.bias_add(fconv4,fval5)
fine4=tf.nn.relu(fbias4)

files=glob.glob(path)
for f in files:
    with Image.open(f) as image:
        start = time.time()
        img2 = image.resize((304,228), Image.ANTIALIAS)
        pixels = np.asarray(img2)
        fineo=sess.run(fine4,feed_dict={input_tensor: pixels.reshape(1, 228, 304, 3)})
        fineo=fineo.reshape([55,74,1])
        depth = fineo.transpose([2, 0, 1])
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        print(process.memory_info()[0])
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")

        output_path=f+'_out.png'
        depth_pil.save(output_path)

        print("time:",time.time() - start,"seconds per frame")
