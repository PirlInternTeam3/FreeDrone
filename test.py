# coding=utf-8

# imports the Pygame library
import pygame
from pyparrot.Bebop import Bebop
import math


def main():
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

    bebop.safe_takeoff(5)

    # print("Flying direct: going forward (positive pitch)") #전진
    # bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.1)
    #
    # print("Flying direct: yaw") #회전
    # bebop.fly_direct(roll=0, pitch=0, yaw=20, vertical_movement=0, duration=0.1)
    #
    # print("Flying direct: going backwards (negative pitch)") #후진
    # bebop.fly_direct(roll=0, pitch=-20, yaw=0, vertical_movement=0, duration=3)
    #
    # print("Flying direct: roll") #좌우
    # bebop.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
    #
    # print("Flying direct: going up") #업
    # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=10, duration=0.1)

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
    bebop.safe_land(5)

    print("DONE - disconnecting")
    bebop.disconnect()


if __name__ == '__main__':
    main()
