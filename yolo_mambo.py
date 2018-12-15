import cv2
import tensorflow as tf
from PIL import Image
from core import utils
import time
import numpy as np
from scipy.spatial import distance
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI

import os

import argparse
import glob
import sys



SIZE = [416, 416]
#video_path = "./data/demo_data/road.mp4"
video_path = 0 # use camera
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_gpu_nms.pb",
                                           ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])


height = 360
width = 640
# drone_centroid
drone_centroid = (int(width / 2), int(height / 2))




class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        # print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "./images/mambo/test_image_%06d.png" % self.index
            # uncomment this if you want to write out images every time you get a new one
            # cv2.imwrite(filename, img)
            self.index +=1
            #print(self.index)


def tracking_target(mamboVision, args):

    mambo = args[0]

    if args[1] == 't':
        testFlying = True
    else :
        testFlying = False

    if (testFlying):
        mambo.safe_takeoff(5)

    with tf.Session() as sess:
        while True:
            image = mamboVision.img.copy()

            if image is None:
                print("드론으로부터 전달받은 이미지 없음!")

            else :
                img_resized = np.array(image.resize(size=tuple(SIZE)), dtype=np.float32)
                img_resized = img_resized / 255.
                prev_time = time.time()

                boxes, scores, labels = sess.run(output_tensors,
                                                 feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
                image = utils.draw_boxes(image, boxes, scores, labels, classes, SIZE, show=False)


                curr_time = time.time()
                exec_time = curr_time - prev_time
                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break


    while True:
        mambo_img = mamboVision.img.copy()

        if mousedown:
            cv2.rectangle(mambo_img,
                          (int(boxToDraw[0]), int(boxToDraw[1])),  # point 1
                          (int(boxToDraw[2]), int(boxToDraw[3])),  # point 2
                          [0, 0, 255], PADDING)  # Color B G R 순서 , Thickness
            if RECORD:
                cv2.circle(mambo_img, (int(drawnBox[2]), int(drawnBox[3])), 10, [255, 0, 0], 4)

            # 마우스 좌클릭을 떼고나면
        elif mouseupdown:
            if initialize:
                outputBoxToDraw = tracker.track('Webcam', mambo_img[:, :, ::-1], boxToDraw)
                initialize = False
            else:
                outputBoxToDraw = tracker.track('Webcam', mambo_img[:, :, ::-1])


            # ROI 트래킹 유지
            cv2.rectangle(mambo_img,
                          (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                          (int(outputBoxToDraw[2]), int(outputBoxToDraw[3])),
                          color=[0, 0, 255], thickness=PADDING)

            target_centroid = (int(outputBoxToDraw[0] + (outputBoxToDraw[2] - outputBoxToDraw[0]) / 2),
                               int(outputBoxToDraw[1] + (outputBoxToDraw[3] - outputBoxToDraw[1]) / 2))

            cv2.circle(mambo_img, target_centroid, radius=4, color=[0, 0, 255], thickness=PADDING)
            cv2.circle(mambo_img, drone_centroid, radius=4, color=[255, 0, 0], thickness=PADDING)
            cv2.arrowedLine(mambo_img, drone_centroid, target_centroid,
                            color=[255, 0, 0], thickness=4)

            dst = distance.euclidean(drone_centroid, target_centroid)

            # pitch_rate = 0
            # yaw_rate = 0
            # vertical_rate = 0

            if dst > 10:
                pitch_rate = int(0 / 10)
                yaw_rate = int(dst / 10)
                vertical_rate = int(0 / 10)

                # 우하단
                if drone_centroid[0] <= target_centroid[0] and drone_centroid[1] <= target_centroid[1]:
                    vertical_rate = -vertical_rate

                # 좌하단
                elif drone_centroid[0] > target_centroid[0] and drone_centroid[1] <= target_centroid[1]:
                    yaw_rate = -yaw_rate
                    vertical_rate = -vertical_rate

                # 좌상단
                elif drone_centroid[0] > target_centroid[0] and drone_centroid[1] > target_centroid[1]:
                    yaw_rate = -yaw_rate

            else:
                pitch_rate = int(0 / 10)
                yaw_rate = int(0 / 10)
                vertical_rate = int(0 / 10)

            print("dst: {}, pitch: {}/s, yaw: {}/s, vertical: {}/s".format(dst, pitch_rate, yaw_rate, vertical_rate))

            if (testFlying):
                mambo.fly_direct(roll=0, pitch=pitch_rate, yaw=yaw_rate, vertical_movement=vertical_rate, duration=1)

        cv2.imshow('Webcam', mambo_img)

        keyPressed = cv2.waitKey(1) and 0xFF
        if keyPressed == ord("q"):
            break

        # cv2 종료
    cv2.destroyAllWindows()

    # land
    if (testFlying):
        mambo.safe_land(5)

    # done doing vision demo
    print("Ending the sleep and vision")
    mamboVision.close_video()

    mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()


if __name__ == "__main__":
    # you will need to change this to the address of YOUR mambo
    mamboAddr = "64:E5:99:F7:22:4A"

    # make my mambo object
    # remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
    mambo = Mambo(mamboAddr, use_wifi=True)
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    if (success):
        # get the state information
        print("sleeping")
        mambo.smart_sleep(1)
        mambo.ask_for_state_update()
        mambo.smart_sleep(1)

        print("Preparing to open vision")
        status = input("Input 't' if you want to TAKE OFF or not : ")
        mamboVision = DroneVisionGUI(mambo, is_bebop=False, buffer_size=200,
                                     user_code_to_run=tracking_target, user_args=(mambo, status))
        userVision = UserVision(mamboVision)
        #mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        mamboVision.open_video()