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
import pygame
import os

height = 480
width = 856
input_size = height * width  # height * width
num_classes = 8  # number of classes


class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):

        print("saving picture")
        self.img = self.vision.get_latest_valid_picture()

        if (self.img is not None):
            filename = "./images/bebop2/test_image_%06d.jpg" % self.index
            print("filename:", filename)
            # cv2.imwrite(filename, self.img)
            self.index += 1
        else:
            print("No img...")

def control_and_collect(bebopVision, args):


    # 이 코드를 돌리기 위해선 먼저 인풋 사이즈와 클래스 갯수를 정해줘야 함.

    # 클래스의 갯수(N) 만큼 one-hot encoding 해준다 ( N X N 단위행렬 )
    k = np.zeros((num_classes, num_classes), 'float')
    for i in range(num_classes):
        k[i, i] = 1


    bebop = args[0]

    # initializes Pygame
    pygame.init()

    # sets the window title
    pygame.display.set_caption(u'FreeDrone Data Collecting...')

    # sets the window size
    pygame.display.set_mode((100, 100))

    # repeat key input
    pygame.key.set_repeat(True)

    # 프레임을 카운트 하기 위한 변수 초기화
    saved_frame = 0
    total_frame = 0

    print("Start collecting images...")
    print("Press 'q' to finish...")

    # 스트리밍 시작시간을 알기 위한 변수 초기화
    start = cv2.getTickCount()

    # 빈 numpy 행렬 생성
    X = np.empty((0, input_size))   # X 에는 사진의 1차원 행렬 데이터가 삽입됨. 크기는 가로픽셀 * 세로픽셀
    y = np.empty((0, num_classes))  # y 에는 라벨 데이터가 삽입됨. 크기는 클래스의 수만큼


    # 이륙
    # bebop.safe_takeoff(5)

    # 동작 전 대기 시간
    # bebop.smart_sleep(5)

    if (bebopVision.vision_running):

        file_name = str(int(time.time()))

        dir_img = "images/bebop2/" + file_name
        if not os.path.exists(dir_img):
            os.makedirs(dir_img)

        # q 입력 받았을 때 while 문 탈출을 위한 loop 변수 선언
        loop = True

        # 현재 프레임 카운트 저장
        frame = 1

        while loop:
            # DroneVisionGUI 클래스에서 img 변수를 생성하고 할당해줘서 버퍼의 이미지를 계속 가져온다.
            b_img = bebopVision.img

            # OpenCV 는 이미지를 None 으로 표시하는 버그가 있으므로, 조건문을 삽입해 None 이 아닐 경우에만 제어 및 데이터 수집 실시
            if (b_img is not None):

                # b_img 는 차원이 (480, 856, 3) 인 RGB 이미지 이므로 gray-scale 로 변환해 (480, 856) 으로 바꿔준다.
                # 이 과정을 거쳐야 X 에 크기가 맞아서 삽입될 수 있다.
                gray_image = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)

                # 임시 배열을 만들어 이를 (1, 480 * 856) 차원으로 변환하고, 데이터 타입도 int에서 float 으로 바꿔준다.
                temp_array = gray_image.reshape(1, input_size).astype(np.float32)

                img_filename = "./{}/test_image_{:06d}.jpg".format(dir_img, frame)

                # get input from pilot
                for event in pygame.event.get():

                    # 파이게임으로 부터 입력 이벤트를 받는데,
                    # '눌러짐' 일 경우 아래 조건문 분기에 따라 이미지 배열과 라벨 배열이 누적이 되며 비밥2 드론을 조종한다.
                    if event.type == pygame.KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        # 단일 입력
                        if key_input[pygame.K_UP]:  # 키보드 위 화살표
                            print("Forward")
                            saved_frame += 1
                            X = np.vstack((X, temp_array))  # np.vstack 은 위에서 아래로 순차적으로 쌓이는 스택이다.
                            y = np.vstack((y, k[0]))        # 전진은 N x N 단위 행렬에서 첫번째 행을 부여한다. 즉 [ 1, 0, ... , 0]
                            # bebop.fly_direct(roll=0, pitch=40, yaw=0, vertical_movement=0, duration=0.1)    # 드론 제어 코드 (전진)
                            cv2.imwrite(img_filename, gray_image) # cv2로 gray image 저장


                        elif key_input[pygame.K_DOWN]:
                            print("Backward")
                            saved_frame += 1
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[1]))
                            # bebop.fly_direct(roll=0, pitch=-40, yaw=0, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            saved_frame += 1
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[2]))
                            # bebop.fly_direct(roll=40, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[3]))
                            saved_frame += 1
                            # bebop.fly_direct(roll=-40, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_w]:
                            print("Up")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[4]))
                            saved_frame += 1
                            # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=15, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_s]:
                            print("Down")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[5]))
                            saved_frame += 1
                            # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-15, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_d]:
                            print("Clockwise")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[6]))
                            saved_frame += 1
                            # bebop.fly_direct(roll=0, pitch=0, yaw=100, vertical_movement=-0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_a]:
                            print("Counter Clockwise")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[7]))
                            saved_frame += 1
                            # bebop.fly_direct(roll=0, pitch=0, yaw=-100, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)


                        elif key_input[pygame.K_q]: # q 를 입력하면 break로 for문을 탈출 하고 이후 False로 while문을 탈출
                            print("quit")
                            loop = False
                            break

                        elif key_input[pygame.K_r]: # r 을 입력하면 현재까지 주행 기록을 초기화 한다.
                            print("reset")
                            X = np.empty((0, input_size))
                            y = np.empty((0, num_classes))


                    # if any key is released
                    elif event.type == pygame.KEYUP:
                        # prints on the console the released key
                        print(u'key released')

                frame += 1
                total_frame += 1

        # land
        # bebop.safe_land(5)

        # save data as a numpy file
        dir_dataset = "training_dataset"
        if not os.path.exists(dir_dataset):
            os.makedirs(dir_dataset)
        try:
            np.savez(dir_dataset + '/' + file_name + '.npz', train=X, train_labels=y) # 수집한 데이터의 칼럼명을 주고 npz로 저장한다.
        except IOError as e:
            print(e)

        end = cv2.getTickCount()
        # calculate streaming duration
        print("Streaming duration: , %.2fs" % ((end - start) / cv2.getTickFrequency()))

        print(X.shape)
        print(y.shape)
        print("Total frame: ", total_frame)
        print("Saved frame: ", saved_frame)
        print("Dropped frame: ", total_frame - saved_frame)


        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()


if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)
    bebop.set_picture_format('jpeg')    # 영상 포맷 변경

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=control_and_collect, user_args=(bebop,))
        userVision = UserVision(bebopVision)
        bebopVision.open_video()

    else:
        print("Error connecting to bebop. Retry")
