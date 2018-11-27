import tensorflow as tf
import pandas as pd
import bebop_create_data
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags 함수를 이용, 자주 쓰는 상수들을 정의
flags.DEFINE_integer('image_size', 96, 'input image size.')
flags.DEFINE_integer('image_color', 1, 'color size.')
flags.DEFINE_integer('maxpool_filter_size', 2, 'maxpool_size.')
flags.DEFINE_integer('num_classes', 3, 'num_classes size.')
flags.DEFINE_integer('batch_size', 100, 'input batch_size.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
# flags.DEFINE_float('keep_prob_train', 0.7, 'input keep_prob_train')
# flags.DEFINE_float('keep_prob_test', 1.0, 'input keep_prob_test')
flags.DEFINE_integer('training_epochs', 200, 'input training_epochs')

def Data_preprocess():
    path_test_bp = "./images/test/before_preprocess/"
    path_test_ap = "./images/test/after_preprocess/"
    bebop_create_data.resize_pixel(path_test_bp, path_test_ap + "after_resize_pixel/", 96, 96)
    bebop_create_data.change_color(path_test_bp + "after_resize_pixel/", path_test_ap)
    bebop_create_data.class_test_label(path_test_ap, "./test_data/")

# 텍스트 파일을 CSV 파일로 읽어와 큐에 셔플해서 넣어준다
def Create_queue(text_file_name):
    train_data = pd.read_csv(text_file_name, names=['image'])
    train_images = list(train_data['image'])
    data_queue = tf.train.slice_input_producer([train_images], num_epochs=2, shuffle=True)

    return data_queue

# 이미지 파일명을 받아서 데이터 객체로 저장
def Read_data(data_queue):
    image_directory = data_queue[0]
    image = tf.image.decode_png(tf.read_file(image_directory), channels=FLAGS.image_color)
    # 3차원이면 tf.image.decode_jpeg

    return image

# 데이터 객체들을 배치 사이즈로 묶어서 리턴
def Data_batch(text_file_name, batch_size):
    data_queue = Create_queue(text_file_name)
    image = Read_data(data_queue)
    image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
    batch_image = tf.train.batch([image] , batch_size = batch_size)

    return batch_image

def Convolution_layer_1(image):
    flags.DEFINE_integer('filter_size_1', 3, 'input filter_size_1')
    flags.DEFINE_integer('layer_size_1', 32, 'input layer_size_1')
    flags.DEFINE_integer('stride_1', 1, 'input stride_1')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_1'):
    W1 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_1, FLAGS.filter_size_1, FLAGS.image_color, FLAGS.layer_size_1], stddev=0.01))
    b1 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_1], stddev=0.01))

    L1 = tf.nn.conv2d(image, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu = tf.nn.relu(tf.add(L1, b1))
    #L1_maxpool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L1_dropout = tf.nn.dropout(L1_maxpool, keep_prob)

    return L1_relu

def Convolution_layer_2(previous_layer):
    flags.DEFINE_integer('filter_size_2', 3, 'input filter_size_2')
    flags.DEFINE_integer('layer_size_2', 64, 'input layer_size_2')
    flags.DEFINE_integer('stride_2', 1, 'input stride_2')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_2'):
    W2 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_2, FLAGS.filter_size_2, FLAGS.layer_size_1, FLAGS.layer_size_2], stddev=0.01))
    b2 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_2], stddev=0.01))

    L2 = tf.nn.conv2d(previous_layer, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu = tf.nn.relu(tf.add(L2, b2))
    L2_maxpool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L2_dropout = tf.nn.dropout(L2_maxpool, keep_prob)

    return L2_maxpool

def Convolution_layer_3(previous_layer):
    flags.DEFINE_integer('filter_size_3', 3, 'input filter_size_3')
    flags.DEFINE_integer('layer_size_3', 128, 'input layer_size_3')
    flags.DEFINE_integer('stride_3', 1, 'input stride_3')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_3'):
    W3 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_3, FLAGS.filter_size_3, FLAGS.layer_size_2, FLAGS.layer_size_3], stddev=0.01))
    b3 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_3], stddev=0.01))

    L3 = tf.nn.conv2d(previous_layer, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3_relu = tf.nn.relu(tf.add(L3, b3))
    L3_maxpool = tf.nn.max_pool(L3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L3_dropout = tf.nn.dropout(L3_maxpool, keep_prob)

    return L3_maxpool

def Convolution_layer_4(previous_layer):
    flags.DEFINE_integer('filter_size_4', 3, 'input filter_size_4')
    flags.DEFINE_integer('layer_size_4', 256, 'input layer_size_4')
    flags.DEFINE_integer('stride_4', 1, 'input stride_4')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_4'):
    W4 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_4, FLAGS.filter_size_4, FLAGS.layer_size_3, FLAGS.layer_size_4], stddev=0.01))
    b4 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_4], stddev=0.01))

    L4 = tf.nn.conv2d(previous_layer, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4_relu = tf.nn.relu(tf.add(L4, b4))
    L4_maxpool = tf.nn.max_pool(L4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L4_dropout = tf.nn.dropout(L4_maxpool, keep_prob)

    return L4_maxpool

def Convolution_layer_5(previous_layer):
    flags.DEFINE_integer('filter_size_5', 3, 'input filter_size_5')
    flags.DEFINE_integer('layer_size_5', 512, 'input layer_size_5')
    flags.DEFINE_integer('stride_5', 1, 'input stride_5')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_5'):
    W5 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_5, FLAGS.filter_size_5, FLAGS.layer_size_4, FLAGS.layer_size_5], stddev=0.01))
    b5 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_5], stddev=0.01))

    L5 = tf.nn.conv2d(previous_layer, W5, strides=[1, 1, 1, 1], padding='SAME')
    L5_relu = tf.nn.relu(tf.add(L5, b5))
    L5_maxpool = tf.nn.max_pool(L5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L5_dropout = tf.nn.dropout(L5_maxpool, keep_prob)

    return L5_maxpool

def Convolution_layer_6(previous_layer):
    flags.DEFINE_integer('filter_size_6', 3, 'input filter_size_6')
    flags.DEFINE_integer('layer_size_6', 1024, 'input layer_size_6')
    flags.DEFINE_integer('stride_6', 1, 'input stride_6')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_6'):
    W6 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_6, FLAGS.filter_size_6, FLAGS.layer_size_5, FLAGS.layer_size_6], stddev=0.01))
    b6 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_6], stddev=0.01))

    L6 = tf.nn.conv2d(previous_layer, W6, strides=[1, 1, 1, 1], padding='SAME')
    L6_relu = tf.nn.relu(tf.add(L6, b6))
    L6_maxpool = tf.nn.max_pool(L6_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L5_dropout = tf.nn.dropout(L5_maxpool, keep_prob)

    return L6_maxpool

def Convolution_layer_7(previous_layer):
    flags.DEFINE_integer('filter_size_7', 3, 'input filter_size_7')
    flags.DEFINE_integer('layer_size_7', 2048, 'input layer_size_7')
    flags.DEFINE_integer('stride_7', 1, 'input stride_7')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_5'):
    W7 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_7, FLAGS.filter_size_7, FLAGS.layer_size_6, FLAGS.layer_size_7], stddev=0.01))
    b7 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_7], stddev=0.01))

    L7 = tf.nn.conv2d(previous_layer, W7, strides=[1, 1, 1, 1], padding='SAME')
    L7_relu = tf.nn.relu(tf.add(L7, b7))
    L7_maxpool = tf.nn.max_pool(L7_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L5_dropout = tf.nn.dropout(L5_maxpool, keep_prob)

    return L7_maxpool

def Convolution_layer_8(previous_layer):
    flags.DEFINE_integer('filter_size_8', 3, 'input filter_size_8')
    flags.DEFINE_integer('layer_size_8', 4096, 'input layer_size_8')
    flags.DEFINE_integer('stride_8', 1, 'input stride_8')

    # 텐서보드에 사용하기 위한 tf.name_scope
    # with tf.name_scope('conv_6'):
    W8 = tf.Variable(tf.truncated_normal(
            [FLAGS.filter_size_8, FLAGS.filter_size_8, FLAGS.layer_size_7, FLAGS.layer_size_8], stddev=0.01))
    b8 = tf.Variable(tf.truncated_normal([FLAGS.layer_size_8], stddev=0.01))

    L8 = tf.nn.conv2d(previous_layer, W8, strides=[1, 1, 1, 1], padding='SAME')
    L8_relu = tf.nn.relu(tf.add(L8, b8))
    L8_maxpool = tf.nn.max_pool(L8_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #L5_dropout = tf.nn.dropout(L5_maxpool, keep_prob)

    return L8_maxpool

def Fully_layer_1(previous_layer):
    flags.DEFINE_integer('F_layer_size_1', 6*6*FLAGS.layer_size_5, 'input F_layer_size_1')
    flags.DEFINE_integer('F_output_size_1', 2048, 'input F_output_size_1')

    #with tf.name_scope('f_1'):

    # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
    previous_layer_flat = tf.reshape(previous_layer, [-1, FLAGS.F_layer_size_1])
    F_W1 = tf.get_variable("F_W1", shape=[FLAGS.F_layer_size_1, FLAGS.F_output_size_1], initializer=tf.contrib.layers.xavier_initializer())
    F_b1 = tf.Variable(tf.truncated_normal([FLAGS.F_output_size_1], stddev=0.01))

    F_L1 = tf.add(tf.matmul(previous_layer_flat, F_W1), F_b1)  # F_L1 = previous_layer_flat*F_W1 + F_b1
    F_L1_relu = tf.nn.relu(F_L1)
    F_L1_dropout = tf.nn.dropout(F_L1_relu, keep_prob)

    return F_L1_dropout

def Fully_layer_2(previous_layer):
    flags.DEFINE_integer('F_layer_size_2', FLAGS.F_output_size_1, 'input F_layer_size_2')
    flags.DEFINE_integer('F_output_size_2', 512, 'input F_output_size_2')

    #with tf.name_scope('f_2'):

    F_W2 = tf.get_variable("F_W2", shape=[FLAGS.F_layer_size_2, FLAGS.F_output_size_2], initializer=tf.contrib.layers.xavier_initializer())
    F_b2 = tf.Variable(tf.truncated_normal([FLAGS.F_output_size_2], stddev=0.01))

    F_L2 = tf.add(tf.matmul(previous_layer, F_W2), F_b2)  # F_L1 = previous_layer_flat*F_W1 + F_b1
    F_L2_relu = tf.nn.relu(F_L2)
    F_L2_dropout = tf.nn.dropout(F_L2_relu, keep_prob)

    return F_L2_dropout

def Final_layer(previous_layer):
    flags.DEFINE_integer('Final_layer_size', FLAGS.F_output_size_2, 'input Final_layer_size')
    flags.DEFINE_integer('Final_output_size', FLAGS.num_classes, 'input Final_output_size')

    # with tf.name_scope('final'):

    Final_W = tf.get_variable("Final", shape=[FLAGS.Final_layer_size, FLAGS.Final_output_size], initializer=tf.contrib.layers.xavier_initializer())
    Final_b = tf.Variable(tf.truncated_normal([FLAGS.Final_output_size], stddev=0.01))

    Final_L = tf.add(tf.matmul(previous_layer, Final_W), Final_b)  # F_L2 = previous_layer*F_W2 + F_b2
    Final_L_relu = tf.nn.relu(Final_L)

    return Final_L_relu

########################################################################################################################

def build_model(images):

    # output shape will be (*,96,96,32)
    Layer1 = Convolution_layer_1(images)  # convolutional layer 1
    print("\nshape after Layer1 ", Layer1.get_shape())

    # output shape will be (*,48,48,64)
    Layer2 = Convolution_layer_2(Layer1)  # convolutional layer 2
    print("shape after Layer2 :", Layer2.get_shape())

    # output shape will be (*,24,24,128)
    Layer3 = Convolution_layer_3(Layer2)  # convolutional layer 3
    print("shape after Layer3 :", Layer3.get_shape())

    # output shape will be (*,12,12,256)
    Layer4 = Convolution_layer_4(Layer3)  # convolutional layer 4
    print("shape after Later4 :", Layer4.get_shape())

    # output shape will be (*,6,6,512)
    Layer5 = Convolution_layer_5(Layer4)  # convolutional layer 5
    print("shape after Layer5 :", Layer5.get_shape())

    # output shape will be (*,12,12,1024)
    #Layer6 = Convolution_layer_6(Layer5)  # convolutional layer 6
    #print("shape after Layer6 :", Layer6.get_shape())
    #
    # # output shape will be (*,6,6,2048)
    #Layer7 = Convolution_layer_7(Layer6)  # convolutional layer 7
    #print("shape after Layer7 :", Layer7.get_shape())
    #
    # # output shape will be (*,3,3,4096)
    # Layer8 = Convolution_layer_8(Layer7)  # convolutional layer 8
    # print("shape after Layer8 :", Layer8.get_shape())

    # fully connected layer 1  (6x6x512 inputs -> 2048 outputs)
    # output shape will be (*,2048)
    F_Layer1 = Fully_layer_1(Layer5)
    print("shape after F_Layer1 :", F_Layer1.get_shape())

    # fully connected layer 2  (2048 inputs -> 512 outputs)
    # output shape will be (*,512)
    F_Layer2 = Fully_layer_2(F_Layer1)
    print("shape after F_Layer2 :", F_Layer2.get_shape())

    # final layer (Final 512 inputs -> 5 outputs)
    # output shape will be (*,5)
    Final = Final_layer(F_Layer2)
    print("shape after Final_Layer :", Final.get_shape())

    return Final

    ## drop out
    # 참고 http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    # 트레이닝시에는 keep_prob < 1.0 , Test 시에는 1.0으로 한다.
    #
    # r_dropout = tf.nn.dropout(r_fc2, keep_prob)
    # print("shape after dropout :", r_dropout.get_shape())

if __name__ == "__main__":

    Data_preprocess()
    images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
    keep_prob = tf.placeholder(tf.float32)

    prediction = tf.nn.softmax(build_model(images))

    train_data = pd.read_csv("./test_data/bebop2_test_data.txt", names=['image'])
    train_data.to_csv('./test_data/bebop2_test_dataset.csv', header=False, index=False)
    bebop2_test_dataset = pd.read_csv('test_data/bebop2_test_dataset.csv')

    test_image_batch = Data_batch("./test_data/bebop2_test_data.txt", len(bebop2_test_dataset.index))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:

        saver = tf.train.Saver()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_images = sess.run(test_image_batch)
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, './checkpoint/test/bebop2_ckpt_test_file')
        p_val = sess.run(prediction, feed_dict={images: test_images, keep_prob: 1.0})
        print(p_val)
        path=[]

        for x in range(len(bebop2_test_dataset.index)):
            path.append(sess.run(tf.argmax(p_val[x])))

    forward = path.count("forward")
    turn_left = path.count("turn_left")
    turn_right = path.count("turn_right")

    name =[forward, turn_left, turn_right]

    print("\n number of prediction : [forward, turn_left, turn_right]\n")
    print("\t--->", name)
    cnt=np.argmax(name)
    x = ["forward", "turn_left", "turn_right"]
