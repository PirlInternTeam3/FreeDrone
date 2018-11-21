from pyparrot.Bebop import Bebop

bebop = Bebop()

print("connecting")
success = bebop.connect(10)
print("success:",success)

i=0
bebop.ask_for_state_update()

while True:
    current_battery = bebop.set_user_sensor_callback(Battery,'batteryLevel')
    print("battery:",current_battery)

    if i==100:
        break

    i+=1

print("DONE - disconnecting")
bebop.disconnect()