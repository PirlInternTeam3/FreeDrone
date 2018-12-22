import yolov3_detect
import yolov3_tracking
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI

class Yolnir(object):

    def __init__(self):
        self.pitch = 0
        self.yaw = 0
        self.vertical = 0

    def main(self):
        mamboAddr = "64:E5:99:F7:22:4A"
        # remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
        mambo = Mambo(mamboAddr, use_wifi=True)
        print("trying to connect to mambo now")
        success = mambo.connect(num_retries=3)
        print("connected: %s" % success)
        is_bebop = False

        if (success):
            # get the state information
            print("sleeping")
            mambo.smart_sleep(1)
            mambo.ask_for_state_update()
            mambo.smart_sleep(1)

            # ????
            print("Preparing to open vision")
            mamboVision = DroneVisionGUI(mambo, is_bebop=is_bebop, buffer_size=200,
                                         user_code_to_run=track_target, user_args=(mambo, (self.pitch, self.yaw, self.vertical)))

            # yolov3_detect object
            userVision = UserVision(mamboVision)
            mamboVision.open_video()

            # # loop
            # video_object = server_video.CollectTrainingData(client.Get_Client(), steer)
            # video_object.collect()
            # microphone_object = server_microphone.Microphone(host, port + 2, steer)
            # microphone_object.Run()


if __name__ == '__main__':

    yolnir = Yolnir()
    yolnir.main()

