import cv2
import sys
import numpy as np
from model import NeuralNetwork

input_size = 480 * 856

model = NeuralNetwork()

model.load_model(path = './model_data/test_model1.h5')

while True:
    img_file = input("파일 입력")

    test_img = cv2.imread('./images/bebop2/1543980588/'+img_file+'.jpg')

    gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    if test_img is None:
            print("can not load image: ", test_img)
            sys.exit()

    test_img_array = gray_test_img.reshape(1, 480, 856, 1).astype(np.float32)

    label = model.predict(test_img_array)

    if label == 0:
        direction = 'Forward'
    elif label == 1:
        direction = 'Backward'
    elif label == 2:
        direction = 'Right'
    elif label == 3:
        direction = 'Left'
    elif label == 4:
        direction = 'Up'
    elif label == 5:
        direction = 'Down'
    elif label == 6:
        direction = 'Clockwise'
    elif label == 7:
        direction = 'Counter Clockwise'

    print(direction)
