"""
Flies the bebop in a fairly wide arc.  You want to be sure you have room for this. (it is commented
out but even what is here is still going to require a large space)
Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
import math

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


while True:
    inp = input("input command")

    if inp == "w":
        bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.5)
    if inp == "s":
        bebop.fly_direct(roll=0, pitch=-20, yaw=0, vertical_movement=0, duration=0.5)
    if inp == "p":
        breakwwwwwww



#print("Turning relative")
#bebop.move_relative(0, 0, 0, math.radians(90))

# this works but requires a larger test space than I currently have. Uncomment with care and test only in large spaces!
#print("Flying direct: going around in a circle (yes you can mix roll, pitch, yaw in one command!)")
#bebop.fly_direct(roll=25, pitch=0, yaw=50, vertical_movement=0, duration=5)

bebop.smart_sleep(1)
bebop.safe_land(5)

print("DONE - disconnecting")
bebop.disconnect()