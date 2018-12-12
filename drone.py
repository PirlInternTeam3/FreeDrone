# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')
graph = tf.get_default_graph()
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import UserVision, DroneVisionGUI

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    return output_layers

class YOLODrone(object):
    def __init__(self, manual=True):
        self.key = None
        self.stop = False
        self.mutex = None
        self.manuel = manual
        self.PID = None
        self.boxes = None
        self.update = False
        self.contours = None
        self.boxes_update = False
        self.image = None
        
    def start(self):
        self.bebop = Bebop()
        success = self.bebop.connect(10)
        
        print("sleeping")
        self.bebop.smart_sleep(5)
        self.bebop.ask_for_state_update()
        self.bebop.safe_takeoff(5)
        
        if (success):
            self.bebopVision = DroneVisionGUI(self.bebop, is_bebop=True, user_code_to_run=self.demo_user_code_after_vision_opened, user_args=(self.bebop,))
            self.userVision = UserVision(self.bebopVision)
            self.bebopVision.set_user_callback_function(self.userVision.save_pictures, user_callback_args=None)
            self.bebopVision.open_video()
            
    def demo_user_code_after_vision_opened(self, bebopVision, args):
        self.bebop = args[0]
        print("Vision successfully started!")
        
        if (self.bebopVision.vision_running):
            print("Moving the camera using velocity")
            
            print("Tilting camera")
            self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=5, duration=3)
            
            while True:
                image = self.bebopVision.img
                Width = image.shape[1]
                Height = image.shape[0]
                scale = 0.00392
                classes = None
                
                with open("./yolov3.txt", 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                    
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
                
                blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(get_output_layers(net))
                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = round(box[0])
                    y = round(box[1])
                    w = round(box[2])
                    h = round(box[3])
                    label = str(classes[class_ids[i]])
                    color = COLORS[class_ids[i]]
                    
                    if label == 'person':
                        image_roi = image[y:y + h, x:x + w]
                        
                        # Filter by color red
                        lower_red_1 = np.array([17, 15, 100])
                        upper_red_1 = np.array([50, 56, 200])
                        
                        try:
                            
                            image_roi = cv2.inRange(image_roi, lower_red_1, upper_red_1)
                            
                            # Put on median blur to reduce noise
                            image_roi = cv2.medianBlur(image_roi, 11)
                            
                            # Find contours and decide if hat is one of them
                            _, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                Area = w * h
                                x_centroid = x + (w / 2)
                                y_centroid = y + (h / 2)
                                
                                # cv2.drawContours(image, self.contours, -1, (0, 255, 0), 3)
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                print("Box Values x: %s, y: %s, x+w: %s, y+h: %s" % (x, y, x + w, y + h))
                                print(Area)
                                
                                if Area < 30000:
                                    print("Forward (positive pitch)")
                                    self.bebop.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=0.5)
                                    
                                elif Area > 60000:
                                    print("Backwards (negative pitch)")
                                    self.bebop.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=0.5)
                                    
                                # x divide into three spaces -> 0-285, 285 - 570, 570 - 856 (x:856)
                                # y divide into three spaces -> 0-160, 160 - 320, 320 - 480 (y:480)
                                if x_centroid > 450:  # the object is on the right side (clockwise), move right a little bit
                                    print("Clockwise-Right (positive yaw)")
                                    self.bebop.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=1)
                                    
                                elif x_centroid < 350:  # the object is on the left side (counter-clockwise), move left a little bit
                                    print("Counterclockwise-Left (negative yaw)")
                                    self.bebop.fly_direct(roll=0, pitch=0, yaw=-50, vertical_movement=0, duration=1)
                                    
                        except:
                            print("No Box Values")
                            continue
                        
                cv2.imshow('cam', image)
                l = cv2.waitKey(150)
                
                if l < 0:
                    continue
                else:
                    key = chr(l)
                    if key == "c":
                        break
                    
            print("Finishing demo and stopping vision")
            self.bebopVision.close_video()
            self.bebop.safe_land(5)
            
        # disconnect nicely so we don't need a reboot
        print("disconnecting")
        self.bebop.disconnect()
        
def main():
    drone = YOLODrone(manual=False)
    drone.start()
    
if __name__ == '__main__':
    main()