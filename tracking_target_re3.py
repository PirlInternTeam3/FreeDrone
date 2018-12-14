"""
Demo of the Bebop vision using DroneVisionGUI (relies on libVLC).
multi-threaded approach than DroneVision
It is a different
Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
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


height = 480
width = 856

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

        print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "./images/bebop2/test/test_image_%06d.jpg" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

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


def tracking_target(bebopVision, args):

    global tracker, initialize

    # # 윈도우 이름설정, 리사이즈
    # cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Webcam', width, height)


    cv2.setMouseCallback('Webcam', on_mouse, 0)

    bebop = args[0]

    print("Press 'q' to finish...")

    # 이륙
    bebop.safe_takeoff(5)
    #
    # # 동작 전 대기 시간
    bebop.smart_sleep(20)

    # 초기화
    while True:
        b_img = bebopVision.img.copy()

        if mousedown:
            cv2.rectangle(b_img,
                          (int(boxToDraw[0]), int(boxToDraw[1])),  # point 1
                          (int(boxToDraw[2]), int(boxToDraw[3])),  # point 2
                          [0, 0, 255], PADDING)  # Color B G R 순서 , Thickness
            if RECORD:
                cv2.circle(b_img, (int(drawnBox[2]), int(drawnBox[3])), 10, [255, 0, 0], 4)


        # 마우스 좌클릭을 떼고나면
        elif mouseupdown:
            if initialize:
                outputBoxToDraw = tracker.track('Webcam', b_img[:, :, ::-1], boxToDraw)
                initialize = False
            else:
                outputBoxToDraw = tracker.track('Webcam', b_img[:, :, ::-1])

            # ROI 트래킹 유지
            cv2.rectangle(b_img,
                          (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                          (int(outputBoxToDraw[2]), int(outputBoxToDraw[3])),
                          color=[0, 0, 255], thickness=PADDING)

            target_centroid = (int(outputBoxToDraw[0] + (outputBoxToDraw[2] - outputBoxToDraw[0]) / 2),
                               int(outputBoxToDraw[1] + (outputBoxToDraw[3] - outputBoxToDraw[1]) / 2))

            cv2.circle(b_img, target_centroid, radius=4, color=[0, 0, 255], thickness=PADDING)
            cv2.circle(b_img, drone_centroid, radius=4, color=[255, 0, 0], thickness=PADDING)
            cv2.arrowedLine(b_img, drone_centroid, target_centroid,
                            color=[255, 0, 0], thickness=4)

            dst = distance.euclidean(drone_centroid, target_centroid)

            # print(dst)

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

                # print("dst: {}, pitch: {}/s, yaw: {}/s, vertical: {}/s".format(dst, pitch_rate, yaw_rate, vertical_rate))
                bebop.fly_direct(roll=0, pitch=pitch_rate, yaw=yaw_rate, vertical_movement=vertical_rate, duration=1)


            else:
                pitch_rate = int(0 / 10)
                yaw_rate = int(0 / 10)
                vertical_rate = int(0 / 10)
                # print("dst: {}, pitch: {}/s, yaw: {}/s, vertical: {}/s".format(dst, pitch_rate, yaw_rate, vertical_rate))
                bebop.fly_direct(roll=0, pitch=pitch_rate, yaw=yaw_rate, vertical_movement=vertical_rate, duration=1)

        cv2.imshow('Webcam', b_img)

        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break

    # cv2 종료
    cv2.destroyAllWindows()

    # land
    bebop.safe_land(5)

    print("Finishing demo and stopping vision")
    bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()


if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # make Re3 Tracker
    RECORD = False
    tracker = re3_tracker.Re3Tracker()

    # connect to the bebop
    success = bebop.connect(5)
    bebop.set_picture_format('jpeg')    # 영상 포맷 변경

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=tracking_target, user_args=(bebop,))
        userVision = UserVision(bebopVision)
        #bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        bebopVision.open_video()

    else:
        print("Error connecting to bebop. Retry")
