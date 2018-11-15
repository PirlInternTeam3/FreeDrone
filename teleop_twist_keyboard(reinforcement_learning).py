#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import cv2
from sklearn.cluster import KMeans
import pyscreenshot as ImageGrab
import random
import dqn
from collections import deque
import roslib;

roslib.load_manifest('teleop_twist_keyboard')
import rospy
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import sys, select, termios, tty


def action(act, speed, turn):
    x = act[0]
    y = act[1]
    z = act[2]
    th = act[3]
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    twist = Twist()
    twist.linear.x = x * speed; twist.linear.y = y * speed; twist.linear.z = z * speed;
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th * turn
    pub.publish(twist)
    time.sleep(1)

    x = 0
    y = 0
    z = 0
    th = 0
    twist.linear.x = x * speed; twist.linear.y = y * speed; twist.linear.z = z * speed;
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th * turn
    pub.publish(twist)


def get_state_and_reward(reward):
    image = ImageGrab.grab(bbox=(2165, 25, 3840, 1080))
    image.save("/home/pirl/pangolin.png")
    image = cv2.imread("/home/pirl/pangolin.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    k = 2
    clt = KMeans(n_clusters=k)
    clt.fit(image)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    get_reward = hist[1] * 100 - reward

    state = cv2.imread("/home/pirl/pangolin.png", cv2.IMREAD_GRAYSCALE)
    state = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    image = image.reshape((image.shape[0] * image.shape[1], 1))

    return state, get_reward


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def main():
    moveBindings = {
        'i': (1, 0, 0, 0), 'j': (0, 0, 0, 1), 'l': (0, 0, 0, -1),
        ',': (-1, 0, 0, 0), 't': (0, 0, 1, 0), 'b': (0, 0, -1, 0),
    }
    def move(x, y, z, th, speed, turn):
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        twist = Twist()
        twist.linear.x = x * speed; twist.linear.y = y * speed; twist.linear.z = z * speed;
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th * turn
        pub.publish(twist)
        time.sleep(1)
        x = 0 ; y = 0 ; z = 0; th = 0
        twist.linear.x = x * speed; twist.linear.y = y * speed; twist.linear.z = z * speed;
        twist.angular.x = 0; twist.angular.y = 0;twist.angular.z = th * turn
        pub.publish(twist)
    def takeoff():
        takeoff = rospy.Publisher("ardrone/takeoff", Empty, queue_size=10)
        takeoff.publish()
    def land():
        land = rospy.Publisher("ardrone/land", Empty, queue_size=10)
        land.publish()
    def getKey():
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('teleop_twist_keyboard')
    speed = rospy.get_param("~speed", 0.9)
    turn = rospy.get_param("~turn", 0.5)
    try:
        while (1):
            key = getKey()
            if key in moveBindings.keys():
                x = moveBindings[key][0] ; y = moveBindings[key][1] ; z = moveBindings[key][2] ; th = moveBindings[key][3]
                print ('move')
                move(x, y, z, th, speed, turn)

            elif key == '1':
                print('takeoff')
                takeoff()
            elif key == '2':
                print('land')
                land()
            elif key == '3':
                break
    except:
        print e

    actions_space = [(1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, -1), (-1, 0, 0, 0)]
    # [forward,turn_left, turn_right, backward, up, down]
    max_episodes = 5000
    replay_buffer = deque()
    input_size = 442464
    output_size = len(actions_space)
    dis = 0.9
    REPLAY_MEMORY = 50000

    with tf.Session() as sess:

        def get_copy_var_ops():
            op_holder = []
            src_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
            dest_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="main")

            for src_vars, dest_vars in zip(src_vars, dest_vars):
                op_holder.append(dest_vars.assign(src_vars.value()))
            sess.run(op_holder)

        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()
        get_copy_var_ops()

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            reward = 0
            state, _ = get_state_and_reward(reward)

            while not done:
                if np.random.rand(1) < e:
                    act = random.randrange(0, len(actions_space))
                else:
                    act = np.argmax(mainDQN.predict(state))

                action(actions_space[act], speed, turn)
                next_state, get_reward = get_state_and_reward(reward)
                reward += get_reward

                replay_buffer.append((state, act, get_reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 1000:
                    done = True

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                get_copy_var_ops()


if __name__ == "__main__":
    main()

