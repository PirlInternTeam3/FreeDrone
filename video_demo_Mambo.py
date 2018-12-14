#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils

from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI

# #video_path = "./data/demo_data/road.mp4"
# #video_path = 0 # use camera

class UserVision:

    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            #filename = "test_image_%06d.png" % self.index
            #cv2.imwrite(filename, img)
            self.index +=1

class YOLODrone(object):

    def __init__(self):
        self.contours = None

    def start(self):
        # mamboAddr = "e0:14:d0:63:3d:d0"
        mamboAddr = "88:36:6C:F1:E9:47"

        self.mambo = Mambo(mamboAddr, use_wifi=True)
        success = self.mambo.connect(num_retries=5)

        #        #print("sleeping")
        #        self.mambo.smart_sleep(5)
        #        self.mambo.ask_for_state_update()
        #        self.mambo.safe_takeoff(5)

        if (success):
            print("success")
            self.mamboVision = DroneVisionGUI(self.mambo, is_bebop=False,
                                              user_code_to_run=self.demo_user_code_after_vision_opened,
                                              user_args=(self.mambo,))
            self.userVision = UserVision(self.mamboVision)
            self.mamboVision.set_user_callback_function(self.userVision.save_pictures, user_callback_args=None)
            self.mamboVision.open_video()

    def demo_user_code_after_vision_opened(self, mamboVision, args):
        self.mambo = args[0]
        print("Vision Successfully Started!")

        if (self.mamboVision.vision_running):
            print("Moving the camera using velocity")

            SIZE = [416, 416]
            classes = utils.read_coco_names('./data/coco.names')
            num_classes = len(classes)
            input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                                        "./checkpoint/yolov3_cpu_nms.pb",
                                                                        ["Placeholder:0", "concat_9:0", "mul_9:0"])

            with tf.Session() as sess:
                while True:
                    IMG = self.mamboVision.img
                    image = Image.fromarray(self.mamboVision.img)

                    img_resized = np.array(image.resize(size=tuple(SIZE)), dtype=np.float32)
                    img_resized = img_resized / 255.
                    prev_time = time.time()

                    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
                    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

                    print(labels)
                    print(boxes)

                    try:
                        #label number for person -> 0
                        l=[]
                        b=[]
                        for i in range(len(labels)):
                            if labels[i] == 0:
                                l.append(labels[i])
                                b.append(boxes[i])
                            else:
                                break

                        print("labels: ", labels)
                        print("l: ", l)

                        image, bbox = utils.draw_boxes(image, boxes, scores, l, classes, SIZE, show=False)

                        print("lengths of bbox: ",len(bbox))

                        for j in range(len(bbox)):

                            x = int(bbox[j][0])
                            y = int(bbox[j][1])
                            x_w = int(bbox[j][2])
                            y_h = int(bbox[j][3])

                            print("x: ", x)
                            print("y: ", y)
                            print("x+w: ", x_w)
                            print("y+h: ", y_h)

                            image_roi = IMG[y:y_h, x:x_w]

                            lower_red_1 = np.array([17, 15, 100])
                            upper_red_1 = np.array([50, 56, 200])

                            try:

                                image_roi = cv2.inRange(image_roi, lower_red_1, upper_red_1)

                                # Put on median blur to reduce noise
                                image_roi = cv2.medianBlur(image_roi, 11)

                                # Find contours and decide if hat is one of them
                                _, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_TREE,
                                                                          cv2.CHAIN_APPROX_SIMPLE)

                                if contours:
                                    w = x_w - x
                                    h = y_h - y

                                    Area = w * h
                                    x_centroid = x + (w / 2)
                                    y_centroid = y + (h / 2)

                                    #cv2.drawContours(image, self.contours, -1, (0, 255, 0), 3)
                                    cv2.rectangle(IMG, (x, y), (x + w, y + h), [0,255,0], 2)
                                    print("Box Values x: %s, y: %s, x+w: %s, y+h: %s" % (x, y, x_w, y_h))
                                    print(Area)

                                    if Area < 30000:
                                        print("Forward (positive pitch)")
                                        #self.mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=0.5)

                                    elif Area > 60000:
                                        print("Backwards (negative pitch)")
                                        #self.mambo.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=0.5)

                                    # x divide into three spaces -> 0-285, 285 - 570, 570 - 856 (x:856)
                                    # y divide into three spaces -> 0-160, 160 - 320, 320 - 480 (y:480)
                                    if x_centroid > 450:  # the object is on the right side (clockwise), move right a little bit
                                        print("Clockwise-Right (positive yaw)")
                                        #self.mambo.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=1)

                                    elif x_centroid < 350:  # the object is on the left side (counter-clockwise), move left a little bit
                                        print("Counterclockwise-Left (negative yaw)")
                                        #self.mambo.fly_direct(roll=0, pitch=0, yaw=-50, vertical_movement=0, duration=1)

                            except:
                               print("No Person with Red")
                               continue

                    except:
                        print("No Person to Detect")
                        continue

                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    result = np.asarray(image)
                    info = "time: %.2f ms" %(1000*exec_time)
                    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)

                    #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("cam", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break


def main():
    drone = YOLODrone()
    drone.start()


if __name__ == '__main__':
    main()
