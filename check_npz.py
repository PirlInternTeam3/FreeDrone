from numpy import load
import cv2

data = load('./training_dataset/1543567954.npz')
lst = data.files    # lst = [ 'train', 'train_labels' ]
for item in lst:
    print(item)
    print(data[item])


# # 라벨링 데이터 출력
# for i in range(len(data['train_labels'])):
#     print(data['train_labels'][i])
