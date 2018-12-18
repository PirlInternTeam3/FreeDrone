import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = './cnn/training_labeled_dataset/*.npz'
training_data = glob.glob(path)

height = 360
width = 640


# load data
for single_npz in training_data:
        with np.load(single_npz) as data:
            x = data['train']
            y = data['train_labels']
            y_df = pd.DataFrame(y, columns=['Forward', 'Right', 'Left'])
            print(len(y))
            print(len(y_df[y_df['Forward'] == 1]))
            print(len(y_df[y_df['Right'] == 1]))
            print(len(y_df[y_df['Left'] == 1]))


# #show img and label
# def show_img_label(index):
#
#     plt.imshow(x[index].reshape(height, width))
#
#     label = np.argmax(y[index])
#
#     if label == 0:
#         direction = 'Forward'
#     elif label == 1:
#         direction = 'Right'
#     elif label == 2:
#         direction = 'Left'
#
#     sub_plt_title = str(label) + " : " + direction
#     plt.title(sub_plt_title)
#     plt.show()
#
#
# # 0번~끝까지 이미지 테스트 해보기
#
# for i in range(len(y)):
#     show_img_label(i)
