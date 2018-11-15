import tensorflow as tf

a = tf.constant(10)
b = tf.constant(20)
add = tf.add(a,b)
sess = tf.Session()

print(sess.run(add))