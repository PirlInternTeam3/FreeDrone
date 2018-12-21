import yolov3_model
import cv2
import threading

class Yolnir(object):
    def __init__(self):
        self.yv3 = yolov3_model.Yolov3()
        self.loop = True

    def detect(self, frame):
        img2 = self.yv3.run_model(frame)
        cv2.imshow('test', img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.loop = False

    def run(self):
        yolnir = threading.Thread(target=self.detect, args=())
        yolnir.start()

y = Yolnir()
vid = cv2.VideoCapture(0)
while y.loop:
    return_value, frame = vid.read()
    y.detect(frame)