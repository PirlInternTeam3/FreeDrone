import threading

class Tracking(object):
    def __init__(self, drone, pitch, yaw, vertical):
        self.drone = drone
        self.pitch = pitch
        self.yaw = yaw
        self.vertical = vertical


    def self_flying(self):

        print("taking off!")
        self.drone.safe_takeoff(5)

        print("Flying direct: going forward (positive pitch)")
        self.drone.fly_direct(roll=0, pitch=self.pitch, yaw=self.yaw, vertical_movement=self.vertical, duration=0.1)

        print("landing")
        self.drone.safe_land(5)

        self.drone_vision.close_video()

        self.drone.smart_sleep(5)

        print("disconnect")
        self.drone.disconnect()

    def run(self):
        track_thread = threading.Thread(target=self.self_flying, args=())
        track_thread.start()