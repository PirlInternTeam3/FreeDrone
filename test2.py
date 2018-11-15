# OpenCV 라이브러리 임포트
import cv2

# 이미지 압력
image = cv2.imread("./images/landscape.jpg")

# 원본 이미지를 회색조로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 원본 이미지와 회색조 이미지를 각각 Windows로 출력
cv2.imshow("Landscape", image)
cv2.imshow("Landscape - gray", gray_image)

# ESC 키 입력 시 Windows 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()