from pyparrot.Bebop import Bebop

bebop = Bebop()

print("connecting...")
success = bebop.connect(10)
print("success status:",success)

if (success):
    print("turning on the video")
    bebop.start_video_stream()

    print("sleeping...")
    bebop.smart_sleep(2)

    bebop.ask_for_state_update()

    bebop.smart_sleep(20)

    print("DONE - disconnecting...")
    bebop.stop_video_stream()
    bebop.smart_sleep(5)
    bebop.disconnect()
