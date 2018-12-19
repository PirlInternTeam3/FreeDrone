import cv2
import tensorflow as tf
from PIL import Image
from core import utils
import time
import numpy as np
from scipy.spatial import distance
import threading


#mambo image
HEIGHT = 360
WIDTH = 640

# ##### HUMAN with RED TRACKING #####
drone_centroid = (int(WIDTH / 2), int(HEIGHT / 2)) # drone_centroid
LOWER_RED_RANGE = np.array([17, 15, 100])
UPPER_RED_RANGE = np.array([50, 56, 200])
TARGET_AREA = 25*10e3
LIMIT_AREA = 30*10e3



class Yolo_Dectection():

    def __init__(self, target_list, drone_img):

        self.target_list = target_list
        self.drone_img = drone_img

    def detect_target(self, args):



        SIZE = [416, 416]
        classes = utils.read_coco_names('./data/coco.names')
        input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                                    "./checkpoint/yolov3_gpu_nms.pb",
                                                                    ["Placeholder:0", "concat_10:0", "concat_11:0",
                                                                     "concat_12:0"])

        with tf.Session() as sess:
            while True:
                drone_image = self.drone_img.copy()

                if drone_image is None:
                    print("드론으로부터 전달받은 이미지 없음!")

                else:
                    image = Image.fromarray(drone_image)
                    img_resized = np.array(image.resize(size=tuple(SIZE)), dtype=np.float32)
                    img_resized = img_resized / 255.
                    prev_time = time.time()

                    boxes, scores, labels = sess.run(output_tensors,
                                                     feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})

                    ##################################################
                    # 여기부터 수정
                    target_only_boxes = list()
                    target_only_scores = list()
                    target_only_labels = list()
                    target_bounded_boxes = list()

                    for i in range(len(labels)):
                        if labels[i] in target_list:
                            target_only_boxes.append(boxes[i])
                            target_only_scores.append(scores[i])
                            target_only_labels.append(labels[i])

                            detection_size, original_size = np.array(SIZE), np.array(image.size)
                            ratio = original_size / detection_size
                            xy = list((boxes[i].reshape(2, 2) * ratio).reshape(-1))
                            target_bounded_boxes.append(xy)

                    image = utils.draw_boxes(image, target_only_boxes, target_only_scores, target_only_labels, classes,
                                             SIZE, show=False)

                    result = np.asarray(image)

                    target_centroid = list()

                    for i in target_bounded_boxes:
                        pt1 = (int(i[0]), int(i[1]))
                        pt2 = (int(i[2]), int(i[3]))

                        image_roi = result[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                        try:

                            image_roi = cv2.inRange(image_roi, LOWER_RED_RANGE, UPPER_RED_RANGE)

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

                    cv2.circle(result, drone_centroid, radius=4, color=[255, 0, 0], thickness=2)

                    dst = list()

                    for i in target_centroid:
                        cv2.circle(result, i, radius=4, color=[0, 0, 255], thickness=2)
                        cv2.arrowedLine(result, drone_centroid, i, color=[255, 0, 0], thickness=4)
                        dst.append(distance.euclidean(drone_centroid, i))

                    try:

                        if dst[0] > 10:
                            yaw_rate = int(dst[0] / 20)
                            vertical_rate = int(dst[0] / 20)

                            # 우하단
                            if drone_centroid[0] <= target_centroid[0][0] and drone_centroid[1] <= target_centroid[0][1]:
                                vertical_rate = -vertical_rate

                            # 좌하단
                            elif drone_centroid[0] > target_centroid[0][0] and drone_centroid[1] <= target_centroid[0][1]:
                                yaw_rate = -yaw_rate
                                vertical_rate = -vertical_rate

                            # 좌상단
                            elif drone_centroid[0] > target_centroid[0][0] and drone_centroid[1] > target_centroid[0][1]:
                                yaw_rate = -yaw_rate

                        else:
                            yaw_rate = 0
                            vertical_rate = 0

                        if area > TARGET_AREA:
                            pitch_rate = -int(LIMIT_AREA / area)

                        elif area == TARGET_AREA:
                            pitch_rate = 0

                        elif area < TARGET_AREA:
                            pitch_rate = int(LIMIT_AREA / area)

                        print(
                            "Area: {}, Distance: {}, \nPitch: {} degree/s, Yaw: {} degree/s, Vertical: {} degree/s".format(area,
                                                                                                                           dst,
                                                                                                                           pitch_rate,
                                                                                                                           yaw_rate,
                                                                                                                           vertical_rate))


                    except:
                        pitch_rate = 0
                        yaw_rate = 0
                        vertical_rate = 0
                        pass

                    return pitch_rate, yaw_rate, vertical_rate

                    ##################################################

                    curr_time = time.time()
                    exec_time = curr_time - prev_time
                    info = "time: %.2f ms" % (1000 * exec_time)
                    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)
                    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("result", result)

                    if cv2.waitKey(1) & 0xFF == ord('q'): break

    def run(self) :
        mic_thread = threading.Thread(target=self.detect_target, args=())
        mic_thread.start()