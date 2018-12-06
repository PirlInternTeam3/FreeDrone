import glob
import numpy as np
from matplotlib import pyplot as plt

path = './training_dataset/*.npz'
training_data = glob.glob(path)

height = 480
width = 856

# load data
for single_npz in training_data:
        with np.load(single_npz) as data:
            x = data['train']
            y = data['train_labels']
            print(x)
            print(y)

#show img and label
def show_img_label(index):

    plt.imshow(x[index].reshape(height, width))

    label = np.argmax(y[index])

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

    sub_plt_title = str(label) + " : " + direction
    plt.title(sub_plt_title)
    plt.show()


# 0번~끝까지 이미지 테스트 해보기

for i in range(len(y)):
    show_img_label(i)
