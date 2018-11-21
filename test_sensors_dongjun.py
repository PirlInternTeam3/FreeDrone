from pyparrot.Bebop import Bebop

bebop = Bebop()

battery = bebop.sensors.battery
flying_state = bebop.sensors.flying_state

print("connecting")
success = bebop.connect(10)
print("success:",success)

i=0
bebop.ask_for_state_update()

while True:
    if i%100==0:
        print("battery:",battery)
        print("flying state:",flying_state)

    if i==1000:
        break
    i+=1

print("DONE - disconnecting")
bebop.disconnect()