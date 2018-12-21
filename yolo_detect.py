import yolov3_model
import cv2
import threading
import time

class Yolnir(object):
    def __init__(self):
        self.yv3 = yolov3_model.Yolov3()
        self.loop = True

    def detect(self, frame):

        prev_time = time.time()
        img2 = self.yv3.run_model(frame)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(img2, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("test", img2)

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