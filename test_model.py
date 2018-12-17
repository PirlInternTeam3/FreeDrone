import cv2
import sys
import numpy as np
from model import NeuralNetwork

height = 360
width = 640
input_size = height * width

model = NeuralNetwork()

model.load_model(path = './model_data/test_model1.h5')

while True:
    img_file = input("파일 입력")

    test_img = cv2.imread('./cnn/images/drone/1545047947/'+img_file+'.jpg')

    gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    if test_img is None:
            print("can not load image: ", test_img)
            sys.exit()

    test_img_array = gray_test_img.reshape(1, height, width, 1).astype(np.float32)

    label = model.predict(test_img_array)

    if label == 0:
        direction = 'Forward'
    elif label == 1:
        direction = 'Right'
    elif label == 2:
        direction = 'Left'

    print(direction)
