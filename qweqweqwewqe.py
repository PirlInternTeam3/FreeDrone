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

    def save_pictures(self, args):

        print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "./images/bebop2/test_image_%06d.png" % self.index
            print("filename:", filename)
            cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

def demo_user_code_after_vision_opened(bebopVision, args):
    bebop = args[0]

    print("Vision successfully started!")
    # removed the user call to this function (it now happens in open_video())
    # bebopVision.start_video_buffering()

    # initializes Pygame
    pygame.init()

    # sets the window title
    pygame.display.set_caption(u'Keyboard events')

    # sets the window size
    pygame.display.set_mode((400, 400))

    #
    pygame.key.set_repeat(True)

    bebop = Bebop()

    print("connecting")
    success = bebop.connect(10)
    print(success)

    print("sleeping")
    bebop.smart_sleep(5)

    bebop.ask_for_state_update()

    # takeoff
    bebop.safe_takeoff(5)

    # skipping actually flying for safety purposes indoors - if you want
    # different pictures, move the bebop around by hand
    bebop.smart_sleep(10)

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
                        bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "DOWN":
                        bebop.fly_direct(roll=0, pitch=-20, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "LEFT":
                        bebop.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "RIGHT":
                        bebop.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                    elif key_name == "W":
                        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=10, duration=0.1)
                    elif key_name == "S":
                        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-10, duration=0.1)
                    elif key_name == "A":
                        bebop.fly_direct(roll=0, pitch=0, yaw=-60, vertical_movement=0, duration=0.1)
                    elif key_name == "D":
                        bebop.fly_direct(roll=0, pitch=0, yaw=60, vertical_movement=-0, duration=0.1)
                    print(u'"{}" key pressed'.format(key_name))

                # if any key is released
                elif event.type == pygame.KEYUP:
                    # prints on the console the released key
                    print(u'"{}" key released'.format(key_name))

        # finalizes Pygame
        pygame.quit()

        bebop.smart_sleep(1)

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
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        bebopVision.open_video()

    else:
        print("Error connecting to bebop. Retry")
