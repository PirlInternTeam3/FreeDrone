#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:15:46 2018

@author: pirl
"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
#image = cv2.imread(args.image)
image = cv2.imread("Toulouse.JPG")
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

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
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

cv2.namedWindow('object detection', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('cam', 856, 480)
cv2.imshow('object detection', image)
cv2.waitKey(1) == 27
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()

#%%

hsv = cv2.cvtColor(self.bebopVision.img, cv2.COLOR_BGR2HSV)
image = cv2.medianBlur(hsv, 3)

# Filter by color red
lower_red_1 = np.array([15, 150, 150])
upper_red_1 = np.array([35, 255, 255])

image = cv2.inRange(image, lower_red_1, upper_red_1)

# Put on median blur to reduce noise
image = cv2.medianBlur(image, 11)

# Find contours and decide if hat is one of them
_, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

self.contours = contours


#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:50:19 2018

@author: pirl
"""

#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
#                help = 'path to input image')
#ap.add_argument('-c', '--config', required=True,
#                help = 'path to yolo config file')
#ap.add_argument('-w', '--weights', required=True,
#                help = 'path to yolo pre-trained weights')
#ap.add_argument('-cl', '--classes', required=True,
#                help = 'path to text file containing class names')
#args = ap.parse_args()

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#            try:
#                self.mutex = Lock()
#                #t1 = Thread(target=self.autonomousFlight, args=(448, 448, 98, 0.02, self.labels,))
#                #t2 = Thread(target=self.getYellow, args=())
#                t3 = Thread(target=self.getPersonBoxes, args=())
#                #t1.start()
#                #t2.start()
#                t3.start()
#                #t1.join()
#                #t2.join()
#                t3.join()
#            except:                
#                print("Error: unable to start thread")


    #def getVideoStream(self, img_width=448, img_height=448):
    def getYellow(self, img_width=448, img_height=448):
        while not self.stop:
#            cv2.drawContours(self.bebopVision.img, self.contours, -1, (0, 255, 0), 3)
#            thresh = 0.02
#            self.mutex.acquire()
#            #self.boxes_update = True
#            if self.boxes_update:
#                self.boxes_update = False
#                for b in self.boxes:
#                    max_class = np.argmax(b.probs)
#                    prob = b.probs[max_class]
#                    print(prob)
#                    if (prob > thresh and self.labels[max_class] == "person"):
#                        left = (b.x - b.w / 2.) * img_width
#                        right = (b.x + b.w / 2.) * img_width
#    
#                        top = (b.y - b.h / 2.) * img_height
#                        bot = (b.y + b.h / 2.) * img_height
#    
#                        cv2.rectangle(self.bebopVision.img, (int(left), int(top)), (int(right), int(bot)), (0, 0, 255), 3)
#                        print("yay")
#                    
#            self.mutex.release()
            
            hsv = cv2.cvtColor(self.bebopVision.img, cv2.COLOR_BGR2HSV)
            image = cv2.medianBlur(hsv, 3)

            # Filter by color red
            lower_red_1 = np.array([15, 150, 150])
            upper_red_1 = np.array([35, 255, 255])

            image = cv2.inRange(image, lower_red_1, upper_red_1)

            # Put on median blur to reduce noise
            image = cv2.medianBlur(image, 11)

            # Find contours and decide if hat is one of them
            _, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #print(contours)

            self.contours = contours
            #print(len(self.contours))
            #cv2.drawContours(self.bebopVision.img, self.contours, -1, (0, 255, 0), 3)
            cv2.drawContours(self.bebopVision.img, self.contours, 0, (0, 255, 0), 2)

#            cv2.namedWindow("drone cam", cv2.WINDOW_NORMAL)
#            cv2.resizeWindow('drone cam', 448, 448)
#            cv2.imshow("drone cam", self.bebopVision.img)
#            cv2.waitKey(1) == 27

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        #  measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    #def getBoundingBoxes(self):
    def getPersonBoxes(self):
        #newest = time.time()
        while True:
            #image = cv2.imread(args.image)
            image = self.bebopVision.img
            Width = image.shape[1]
            Height = image.shape[0]
            scale = 0.00392
            
            classes = None
            
            with open("/home/pirl/Desktop/Drone_Merge/yolov3.txt", 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            
            #net = cv2.dnn.readNet(args.weights, args.config)
            net = cv2.dnn.readNet("/home/pirl/Desktop/Drone_Merge/yolov3.weights", "/home/pirl/Desktop/Drone_Merge/yolov3.cfg")
            
            blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
            
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
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                label = str(classes[class_ids[i]])

                color = COLORS[class_ids[i]]
                
                if label == 'person':
                    
                    image_roi = image[round(y):round(y+h), round(x):round(x+w)]
        
                    # Filter by color red
                    #lower_red_1 = np.array([15, 150, 150])
                    #upper_red_1 = np.array([35, 255, 255])
                    lower_red_1 = np.array([17, 15, 100])
                    upper_red_1 = np.array([50, 56, 200])
                    
                    try:
                        image_roi = cv2.inRange(image_roi, lower_red_1, upper_red_1)
            
                        # Put on median blur to reduce noise
                        image_roi = cv2.medianBlur(image_roi, 11)
            
                        # Find contours and decide if hat is one of them
                        _, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            
                            cv2.drawContours(image, self.contours, 0, (0, 255, 0), 2)
                            cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), color, 2)
                            cv2.putText(image, label, (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            print("Box Values x: %s, y: %s, x+w: %s, y+h: %s" %(round(x), round(y), round(x+w), round(y+h)))
                    
                    except:
                        continue

                    print("YYAAAYYY")


#                cv2.namedWindow('object detection', cv2.WINDOW_NORMAL)
#                cv2.resizeWindow('object detection', 856, 480)
                    
            cv2.imshow('cam', image)
            cv2.waitKey(1) == 27        
        
#        while not self.stop:
#            pixelarray = self.bebopVision.img
#            pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)
#            # Check for Blurry
#            gray = cv2.cvtColor(pixelarray, cv2.COLOR_RGB2GRAY)
#            fm = self.variance_of_laplacian(gray)
#            
#            if fm < 10:
#                continue
#
#            ima = cv2.resize(pixelarray, (448, 448))
#
#            image = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)
#
#            image = np.rollaxis(image, 2, 0)
#            image = image / 255.0
#            image = image * 2.0 - 1.0
#            image = np.expand_dims(image, axis=0)
#
#            global graph
#            with graph.as_default():
#                out = self.model.predict(image)
#            
#            predictions = out[0]
#            print(predictions)
#            boxes = convert_yolo_detections(predictions)
#            print(boxes)
#
#            self.mutex.acquire()
#            self.boxes = do_nms_sort(boxes, 98)
#            self.image = ima
#            self.update = True
#            self.mutex.release()

#            except:
#                 pass


#    def getKeyInput(self):
#        while not self.stop:  # while 'bedingung true'
#            time.sleep(0.1)
#
#
#            if self.key == "t":  # if 'bedingung true'
#                self.drone.takeoff()
#            elif self.key == " ":
#                self.drone.land()
#            elif self.key == "0":
#                self.drone.hover()
#            elif self.key == "w":
#                self.drone.move_forward()
#            elif self.key == "s":
#                self.drone.move_backward()#                self.drone.turn_right()
#            elif self.key == "8":
#                self.drone.move_up()
#            elif self.key == "2":
#                self.drone.move_down()
#            elif self.key == "c":
#                self.stop = True
#            elif self.key == "a":
#                self.drone.move_left()
#            elif self.key == "d":
#                self.drone.move_right()
#            elif self.key == "q":
#                self.drone.turn_left()
#            elif self.key == "e":
#                self.drone.turn_right()
#            elif self.key == "8":
#                self.drone.move_up()
#            elif self.key == "2":
#                self.drone.move_down()
#            elif self.key == "c":
#                self.stop = True
#            else:
#                self.drone.hover()
#
#            if self.key != " ":
#                self.key = ""

    def autonomousFlight(self, img_width, img_height, num, thresh, labels):
        #actuator = Actuator(self.drone, img_width, img_width * 0.5)

        #print self.drone.navdata
        while not self.stop:
            if self.update == True:
                self.update = False

                hsv = cv2.cvtColor(self.bebopVision.img, cv2.COLOR_BGR2HSV)
                image = cv2.medianBlur(hsv, 3)

                # Filter by color red
                lower_red_1 = np.array([15, 150, 150])
                upper_red_1 = np.array([35, 255, 255])

                image = cv2.inRange(image, lower_red_1, upper_red_1)

                # Put on median blur to reduce noise
                image = cv2.medianBlur(image, 11)

                # Find contours and decide if hat is one of them
                _, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                self.contours = contours
                # print(self.contours)
#
                boxes = self.boxes

                best_prob = -99999
                best_box = -1
                best_contour = None

                self.mutex.acquire()
                
#                for i in range(num):
#                    # for each box, find the class with maximum prob
#                    max_class = np.argmax(boxes[i].probs)
#                    prob = boxes[i].probs[max_class]
#                    temp = boxes[i].w
#                    boxes[i].w = boxes[i].h
#                    boxes[i].h = temp
#
#                    if prob > thresh and labels[max_class] == "person":
#                        for contour in contours:
#                            x, y, w, h = cv2.boundingRect(contour)
#
#                            left = (boxes[i].x - boxes[i].w / 2.) * img_width
#                            right = (boxes[i].x + boxes[i].w / 2.) * img_width
#
#                            top = (boxes[i].y - boxes[i].h / 2.) * img_height
#                            bot = (boxes[i].y + boxes[i].h / 2.) * img_height
#
#                            if not (x + w < left or right < x or y + h < top or bot < y):
#                               if best_prob < prob and w > 30:
#                                    print("prob found")
#                                    best_prob = prob
#                                    best_box = i
#                                    best_contour = contour
#                                    #print(best_contour)
#
#                self.boxes_update = True
                
#                if best_box < 0:
#                    # print "No Update"
#                    self.mutex.release()
#                    self.drone.at(libardrone.at_pcmd, False, 0, 0, 0, 0)
#                    continue

#                b = boxes[best_box]
#
#                left = (b.x - b.w / 2.) * img_width
#                right = (b.x + b.w / 2.) * img_width
#
#                top = (b.y - b.h / 2.) * img_height
#                bot = (b.y + b.h / 2.) * img_height
#
#                if (left < 0): left = 0;
#                if (right > img_width - 1): right = img_width - 1;
#                if (top < 0): top = 0;
#                if (bot > img_height - 1): bot = img_height - 1;
#
#                width = right - left
#                height = bot - top
#                x, y, w, h = cv2.boundingRect(best_contour)
                #cv2.drawContours(self.bebopVision.img, best_contour, -1, (0, 255, 0), 3)
                #actuator.step(right - width/2., width)
                self.mutex.release()

def main():
    drone = YOLODrone(manual=False)
    drone.start()

if __name__ == '__main__':
    main()
