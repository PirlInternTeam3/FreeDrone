from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI
import cv2

import time
import numpy as np
import os

import argparse
import glob
import sys
from scipy.spatial import distance

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

from tracker import re3_tracker

from re3_utils.util import drawing
from re3_utils.util import bb_util
from re3_utils.util import im_util

# set this to true if you want to fly for the demo

height = 360
width = 640

# Tracking을 위한 변수선언
PADDING = 2
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

# drone_centroid
drone_centroid = (int(width / 2), int(height / 2))

drawnBox = np.zeros(4) # 4짜리 벡터
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False


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


def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN: # 좌클릭시
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])


def tracking_target(mamboVision, args):
    """
    Demo the user code to run with the run button for a mambo
    :param args:
    :return:
    """

    mambo = args[0]

    if args[1] == 't':
        testFlying = True
    else :
        testFlying = False

    global tracker, initialize

    cv2.setMouseCallback('Webcam', on_mouse, 0)

    if (testFlying):
        mambo.safe_takeoff(5)

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

    # make Re3 Tracker
    RECORD = False
    tracker = re3_tracker.Re3Tracker()

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
        status = input("input 't' if you want to TAKE OFF or not")
        mamboVision = DroneVisionGUI(mambo, is_bebop=False, buffer_size=200,
                                     user_code_to_run=tracking_target, user_args=(mambo, status))
        userVision = UserVision(mamboVision)
        #mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        mamboVision.open_video()