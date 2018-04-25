# Walt: malphite99!, El: giraffe123, Jazz: chocojazz!, Brian: bfrieds01
# to install pynput: pip install pynput
# how to run: sudo python keylogger.py > [Your Name]_[Your Password].csv
# end it with a CTRL-C

from pynput.keyboard import Key, Listener
import time
import csv

from classifier import train_classifier, classify

PRESS = "press"
RELEASE = "release"

clf = train_classifier()

averages = {}
prev = None
numbers = { "\'" + str(num) + "\'" for num in range(10) }
delta_lim = 2 # in seconds

def on_press(key):
    log(key)
    classified = classify(clf, averages)
    if classified:
        print(classified)

def log(key):
    global prev
    print("hey")
    if key in numbers:
        pass
    elif prev:
        currTime = time.time()
        time_delta = currTime - prev[1]
        if time_delta < delta_lim:
            combo = str(prev[0]) + "," + str(key)
            if combo in averages:
                average, count = averages[combo]
                averages[combo] = ((average * count + time_delta) / (count + 1) , count + 1)
            else:
                averages[combo] = (time_delta, 1)
        prev = (key, currTime)
    else:
        prev = (key, time.time())

with Listener(on_press=on_press) as listener:
    listener.join()