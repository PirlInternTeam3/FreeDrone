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

# https://opencv-python.readthedocs.io/en/latest/doc/16.imageContourFeature/imageContourFeature.html

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils


SIZE = [416, 416]
#video_path = "./data/demo_data/road.mp4"
video_path = 0 # use camera
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_gpu_nms.pb",
                                           ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])


#추가한 변수
TARGET = 0 # person
HEIGHT = 480
WIDTH = 640
PADDING = 2
drone_centroid = (int(WIDTH / 2), int(HEIGHT / 2))
LOWER_RED_RANGE = np.array([17, 15, 100])
UPPER_RED_RANGE = np.array([50, 56, 200])

with tf.Session() as sess:
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        img_resized = np.array(image.resize(size=tuple(SIZE)), dtype=np.float32)
        img_resized = img_resized / 255
        prev_time = time.time()


        boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        # image = utils.draw_boxes(image, boxes, scores, labels, classes, SIZE, show=False)

        ##################################################
        #여기부터 수정
        target_only_boxes = list()
        target_only_scores = list()
        target_only_labels = list()
        target_bounded_boxes = list()

        for i in range(len(labels)):
            if labels[i] == TARGET: # PERSON
                target_only_boxes.append(boxes[i])
                target_only_scores.append(scores[i])
                target_only_labels.append(labels[i])

                detection_size, original_size = np.array(SIZE), np.array(image.size)
                ratio = original_size / detection_size
                xy = list((boxes[i].reshape(2, 2) * ratio).reshape(-1))
                target_bounded_boxes.append(xy)




        image = utils.draw_boxes(image, target_only_boxes, target_only_scores, target_only_labels, classes, SIZE, show=False)

        result = np.asarray(image)
        ##################################################


        target_centroid = list()

        ###############
        for i in target_bounded_boxes:
            pt1 = (int(i[0]), int(i[1]))
            pt2 = (int(i[2]), int(i[3]))

            image_roi = result[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            try:
                # 해당 영역에서 Red color를 찾는다.
                image_roi = cv2.inRange(image_roi, LOWER_RED_RANGE, UPPER_RED_RANGE)

                if image_roi is not None:
                    # 노이즈를 줄이기 위해 블러링 한다
                    image_roi = cv2.medianBlur(image_roi, 11)

                    cv2.imshow('ROI', image_roi)

                    # Find contours and decide if hat is one of them
                    # find contours 함수는 원본 이미지를 직접 수정하기 때문에 원본이미지 보존을 위해 .copy 사용한다.
                    # cv2.RETR_TREE 는 모든 contours line을 찾으며, 모든 hieracy관계를 구성함.
                    # contour를 찾을때 근사치 찾는 방법으로 APPROX_SIMPLE은 contours line을 그릴 수 있는 point 만 저장
                    # Returns:	image, contours , hierachy

                    #img_contours, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    target_centroid.append((int(pt1[0] + dx / 2), int(pt1[1] + dy / 2)))

                    cv2.rectangle(result, pt1=pt1, pt2=pt2, color=[0, 0, 255], thickness=PADDING*3)
                    # cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 5)
                    # area = cv2.contourArea(cnt)
                    # print(area)

                    #######################
                    # # issue 1
                    # imgray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
                    # # threshold를 이용하여 binary image로 변환
                    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                    # cv2.imshow('thresh_cam', thresh)
                    # # 해당 영역에서 Red color를 찾는다.
                    # image_roi = cv2.inRange(image_roi, LOWER_RED_RANGE, UPPER_RED_RANGE)
                    # # 노이즈를 줄이기 위해 블러링 한다
                    # image_roi = cv2.medianBlur(image_roi, 11)
                    # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # cv2.drawContours(result, contours, -1, (0, 0, 255), 5)
                    #######################
                else:
                    print("Nope")

            except:
                print("No Person with Red")
                continue

        ##############




        # ##############
        # # centroid 추가
        #
        for i in target_centroid:
            cv2.circle(result, i, radius=4, color=[0, 0, 255], thickness=PADDING)
            cv2.arrowedLine(result, drone_centroid, i, color=[255, 0, 0], thickness=4)

        cv2.circle(result, drone_centroid, radius=4, color=[255, 0, 0], thickness=PADDING)
        # #############


        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


