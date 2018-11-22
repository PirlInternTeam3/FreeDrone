from pyparrot.Bebop import Bebop

bebop = Bebop()
sensor_dict=bebop.sensors.sensors_dict


print("connecting")
success = bebop.connect(10)
print("connection success:",success)

bebop.ask_for_state_update()

# for key, value in sensors_dict.items():
#     print(key,":",value)

while True:
    print("BatteryStateChanged_battery_percent:",sensor_dict['BatteryStateChanged_battery_percent'])
    print("AttitudeChanged_roll:",sensor_dict['AttitudeChanged_roll'])
    print("AttitudeChanged_pitch:",sensor_dict['AttitudeChanged_pitch'])
    print("AttitudeChanged_yaw:", sensor_dict['AttitudeChanged_yaw'])

    print("")
    print("")

print("DONE - disconnecting")
bebop.disconnect()