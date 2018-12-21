import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
from scipy.spatial import distance


class Yolov3(object):
    def __init__(self):

        self.SIZE = [416, 416]
        self.classes = utils.read_coco_names('./data/coco.names')
        self.num_classes = len(self.classes)
        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_gpu_nms.pb",
                                                   ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
        self.sess = tf.Session()
        self.LOWER_RED_RANGE = np.array([17, 15, 100])
        self.UPPER_RED_RANGE = np.array([50, 56, 200])
        self.pitch_rate = 0
        self.yaw_rate = 0
        self.vertical_rate = 0
        self.TARGET = [0]
        self.drone_centroid = (int(640 / 2), int(480 / 2)) # drone_centroid


    def run_model(self, frame):
        if frame is None:
            print("No image!")
        else:
            prev_time = time.time()

            #####TF MODEL#####
            image = Image.fromarray(frame)
            img_resized = np.array(image.resize(size=tuple(self.SIZE)), dtype=np.float32)
            img_resized = img_resized / 255
            boxes, scores, labels = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: np.expand_dims(img_resized, axis=0)})
            image, bbox_resized = utils.draw_boxes(image, boxes, scores, labels, self.classes, self.SIZE, show=False, target=self.TARGET)
            result = np.asarray(image)
            self.pitch_rate, self.yaw_rate, self.vertical_rate, final_img = self.calculate_pitch_yaw_vertical(bbox_resized, result)

        return final_img

    def calculate_pitch_yaw_vertical(self, bbox_resized, result):

        target_centroid = list()

        for i in bbox_resized:
            pt1 = (int(i[0]), int(i[1]))
            pt2 = (int(i[2]), int(i[3]))

            image_roi = result[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            try:

                image_roi = cv2.inRange(image_roi, self.LOWER_RED_RANGE, self.UPPER_RED_RANGE)

                image_roi = cv2.medianBlur(image_roi, 11)

                cv2.imshow('ROI', image_roi)

                _, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    target_centroid.append((int(pt1[0] + dx / 2), int(pt1[1] + dy / 2)))
                    cv2.rectangle(result, pt1=pt1, pt2=pt2, color=[0, 0, 255], thickness=6)
                    area = dx * dy

            except:
                print("No Person with Red")
                continue

        cv2.circle(result, self.drone_centroid, radius=4, color=[255, 0, 0], thickness=2)

        dst = list()

        for i in target_centroid:
            cv2.circle(result, i, radius=4, color=[0, 0, 255], thickness=2)
            cv2.arrowedLine(result, self.drone_centroid, i, color=[255, 0, 0], thickness=4)
            dst.append(distance.euclidean(self.drone_centroid, i))

        try:

            if dst[0] > 10:
                yaw_rate = int(dst[0] / 20)
                vertical_rate = int(dst[0] / 20)

                # 우하단
                if self.drone_centroid[0] <= target_centroid[0][0] and self.drone_centroid[1] <= target_centroid[0][1]:
                    vertical_rate = -vertical_rate

                # 좌하단
                elif self.drone_centroid[0] > target_centroid[0][0] and self.drone_centroid[1] <= target_centroid[0][1]:
                    yaw_rate = -yaw_rate
                    vertical_rate = -vertical_rate

                # 좌상단
                elif self.drone_centroid[0] > target_centroid[0][0] and self.drone_centroid[1] > target_centroid[0][1]:
                    yaw_rate = -yaw_rate

            else:
                yaw_rate = 0
                vertical_rate = 0

            if area > 25000:
                pitch_rate = -int(area / 30000)

            else:
                pitch_rate = int(30000 / area)

        except:
            pass

        ##################################################

        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
        ##################################################

        return result

