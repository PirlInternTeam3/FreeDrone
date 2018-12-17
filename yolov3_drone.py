import cv2
import tensorflow as tf
from PIL import Image
from core import utils
import time
import numpy as np

from scipy.spatial import distance

from pyparrot.Minidrone import Mambo
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI

#추가한 변수
TARGET = 0 # person

# #bebop image
# HEIGHT = 480
# WIDTH = 856

#mambo image
HEIGHT = 360
WIDTH = 640

PADDING = 2
# drone_centroid
drone_centroid = (int(WIDTH / 2), int(HEIGHT / 2))
LOWER_RED_RANGE = np.array([17, 15, 100])
UPPER_RED_RANGE = np.array([50, 56, 200])



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


def tracking_target(droneVision, args):

    drone = args[0]

    if args[1] == 't':
        testFlying = True
    else :
        testFlying = False

    if (testFlying):
        drone.safe_takeoff(5)

    SIZE = [416, 416]
    classes = utils.read_coco_names('./data/coco.names')
    input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                                "./checkpoint/yolov3_gpu_nms.pb",
                                                                ["Placeholder:0", "concat_10:0", "concat_11:0",
                                                                 "concat_12:0"])

    with tf.Session() as sess:
        while True:
            drone_image = droneVision.img.copy()

            if drone_image is None:
                print("드론으로부터 전달받은 이미지 없음!")

            else :
                image = Image.fromarray(drone_image)
                img_resized = np.array(image.resize(size=tuple(SIZE)), dtype=np.float32)
                img_resized = img_resized / 255.
                prev_time = time.time()

                boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})

                ##################################################
                # 여기부터 수정
                target_only_boxes = list()
                target_only_scores = list()
                target_only_labels = list()
                target_bounded_boxes = list()

                for i in range(len(labels)):
                    if labels[i] == TARGET:  # PERSON
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
                            cv2.rectangle(result, pt1=pt1, pt2=pt2, color=[0, 0, 255], thickness=PADDING * 3)
                            area = dx * dy

                    except:
                        print("No Person with Red")
                        continue

                cv2.circle(result, drone_centroid, radius=4, color=[255, 0, 0], thickness=PADDING)

                dst = list()

                for i in target_centroid:
                    cv2.circle(result, i, radius=4, color=[0, 0, 255], thickness=PADDING)
                    cv2.arrowedLine(result, drone_centroid, i, color=[255, 0, 0], thickness=4)
                    dst.append(distance.euclidean(drone_centroid, i))

                try:

                    if dst[0] > 10:
                        yaw_rate = int(dst[0] / 10)
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


                    if area > 25*10e3:
                        pitch_rate = -int(35*10e3 / area)

                    elif area == 25*10e3:
                        pitch_rate = 0

                    elif area < 25*10e3:
                        pitch_rate = int(35*10e3 / area)

                    print("Area: {}, Distance: {}, \nPitch: {} degree/s, Yaw: {} degree/s, Vertical: {} degree/s".format(area, dst, pitch_rate, yaw_rate, vertical_rate))

                    if (testFlying):
                        drone.fly_direct(roll=0, pitch=pitch_rate, yaw=yaw_rate, vertical_movement=vertical_rate,
                                         duration=0.1)

                except:
                    if (testFlying):
                        drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=1)
                    pass

                ##################################################

                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time: %.2f ms" % (1000 * exec_time)
                cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", result)



                if cv2.waitKey(1) & 0xFF == ord('q'): break

    # land
    if (testFlying):
        drone.safe_land(5)

    # done doing vision demo
    print("Ending the sleep and vision")
    droneVision.close_video()

    drone.smart_sleep(5)

    print("disconnecting")
    drone.disconnect()


if __name__ == "__main__":

    drone_type = input("Input drone type 'bebop' or 'mambo : ")

    if drone_type == 'bebop':
        drone = Bebop()
        success = drone.connect(5)
        drone.set_picture_format('jpeg')  # 영상 포맷 변경
        is_bebop = True
    elif drone_type =='mambo':
        mamboAddr = "64:E5:99:F7:22:4A"
        drone = Mambo(mamboAddr, use_wifi=True)
        success = drone.connect(num_retries=3)
        is_bebop = False
        #drone.set_max_tilt()
    if (success):
        # get the state information
        print("sleeping")
        drone.smart_sleep(1)
        drone.ask_for_state_update()
        drone.smart_sleep(1)
        print("Preparing to open vision")

        status = input("Input 't' if you want to TAKE OFF or not : ")
        droneVision = DroneVisionGUI(drone, is_bebop=is_bebop, buffer_size=200, user_code_to_run=tracking_target, user_args=(drone,status))
        userVision = UserVision(droneVision)
        #bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        droneVision.open_video()