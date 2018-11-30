# OpenCV 라이브러리 임포트
import cv2
import numpy as np

# 이미지 압력
img = cv2.imread("./images/bebop2/test_image_000000.jpg")
img_str = cv2.imencode('.jpg', img)[1].tostring()
image = cv2.imdecode(np.frombuffer(img_str, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# 원본 이미지를 회색조로 변환
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 원본 이미지와 회색조 이미지를 각각 Windows로 출력
cv2.imshow('origin', img)
# cv2.imshow("Landscape - gray", gray_image)

# ESC 키 입력 시 Windows 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()


'''Finding jpeg image in a binary file
first = stream_bytes.find(b'\xff\xd8') # start of image
last = stream_bytes.find(b'\xff\xd9') # end of image
'''