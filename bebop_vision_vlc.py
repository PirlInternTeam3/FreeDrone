"""
Demo of the Bebop ffmpeg based vision code (basically flies around and saves out photos as it flies)
Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time

