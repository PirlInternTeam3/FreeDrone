"""
Demo of the Bebop vision using DroneVisionGUI (relies on libVLC).
multi-threaded approach than DroneVision
It is a different
Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time
import pygame

isAlive = False

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures_forward(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/forward/forward_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

    def save_pictures_backward(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/backward/backward_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

    def save_pictures_move_left(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/move_left/move_left_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

    def save_pictures_move_right(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/move_right/move_right_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

    def save_pictures_turn_left(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/turn_left/turn_left_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

    def save_pictures_turn_right(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            filename = "./images/turn_right/turn_right_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

def demo_user_code_after_vision_opened(bebopVision, args):

    # initializes Pygame
    pygame.init()

    # sets the window title
    pygame.display.set_caption(u'Keyboard events')

    # sets the window size
    pygame.display.set_mode((400, 400))

    # repeat key input
    pygame.key.set_repeat(True)


    bebop = args[0]

    print("Vision successfully started!")
    # removed the user call to this function (it now happens in open_video())
    # bebopVision.start_video_buffering()

    # takeoff
    bebop.safe_takeoff(5)

    # skipping actually flying for safety purposes indoors - if you want
    # different pictures, move the bebop around by hand
    print("Fly me around by hand!")
    bebop.smart_sleep(5)

    if (bebopVision.vision_running):

        # infinite loop
        while True:
            # gets a single event from the event queue
            event = pygame.event.wait()

            # if the 'close' button of the window is pressed
            if event.type == pygame.QUIT:
                # stops the application
                break

            # cptures the 'KEYDOWN' and 'KEYUP' events
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                # gets the key name
                key_name = pygame.key.name(event.key)

                # converts to uppercase the key name
                key_name = key_name.upper()

                # if any key is pressed
                if event.type == pygame.KEYDOWN:
                    # prints on the console the key pressed
                    if key_name == "UP":
                        print()
                        userVision.save_pictures_forward(args)
                        bebop.fly_direct(roll=0, pitch=40, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "DOWN":
                        print()
                        userVision.save_pictures_backward(args)
                        bebop.fly_direct(roll=0, pitch=-40, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "LEFT":
                        print()
                        userVision.save_pictures_move_left(args)
                        bebop.fly_direct(roll=-40, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "RIGHT":
                        print()
                        userVision.save_pictures_move_right(args)
                        bebop.fly_direct(roll=40, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "W":
                        print()
                        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=15, duration=0.1)
                    elif key_name == "S":
                        print()
                        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-15, duration=0.1)
                    elif key_name == "A":
                        print()
                        userVision.save_pictures_turn_left(args)
                        bebop.fly_direct(roll=0, pitch=0, yaw=-100, vertical_movement=0, duration=0.1)
                    elif key_name == "D":
                        print()
                        userVision.save_pictures_turn_right(args)
                        bebop.fly_direct(roll=0, pitch=0, yaw=100, vertical_movement=-0, duration=0.1)
                    elif key_name == "Q":
                        print()
                        #land command
                        break
                    print(u'"{}" key pressed'.format(key_name))


                # if any key is released
                elif event.type == pygame.KEYUP:
                    # prints on the console the released key
                    print(u'"{}" key released'.format(key_name))

        # finalizes Pygame
        pygame.quit()

        bebop.smart_sleep(5)

        # land
        bebop.safe_land(5)

        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()


if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=demo_user_code_after_vision_opened, user_args=(bebop,))
        userVision = UserVision(bebopVision)
        bebopVision.open_video()

    else:
        print("Error connecting to bebop. Retry")
