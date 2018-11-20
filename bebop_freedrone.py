from pyparrot.Bebop import Bebop

bebop = Bebop()

print("connecting")
success = bebop.connect(10)
print(success)

if (success):
    print("turning on the video")
    bebop.start_video_stream()
    bebop.set_video_stream_mode()

    print("sleeping")

    bebop.smart_sleep(5)

    bebop.ask_for_state_update()

    bebop.safe_takeoff(5)

    # set safe indoor parameters
    bebop.set_max_tilt(5)
    bebop.set_max_vertical_speed(1)

    while True:
        inp = input("input command")


        if inp == "w":
            bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.5)
        if inp == "s":
            bebop.fly_direct(roll=0, pitch=-20, yaw=0, vertical_movement=0, duration=0.5)
        if inp == "p":
            break


    bebop.smart_sleep(5)

    bebop.safe_land(10)

    print("DONE - disconnecting")
    bebop.stop_video_stream()
    bebop.smart_sleep(5)
    print(bebop.sensors.battery)
    bebop.disconnect()