from pynput.keyboard import Key, Listener
import time
import csv

PRESS = "press"
RELEASE = "release"
FILENAME = "walt_bfrieds01.csv"
RET_FILENAME = "walt_bfrieds01_clean.csv"
out = []
delta_lim = 0.5 # in seconds

numbers = { "\'" + str(num) + "\'" for num in range(10) }

with open(FILENAME, 'r') as csvfile:
	reader = csv.reader(csvfile)
	prev = None
	for row in reader:
		if row[2] == PRESS:
			if row[0] in numbers:
				continue
			elif prev:
				time_delta = float(row[1]) - float(prev[1])
				if time_delta < delta_lim:
					combo = prev[0] + "," + row[0]
					out.append((combo, time_delta))
				prev = row
			else:
				prev = row

with open(RET_FILENAME, 'w') as csvfile:
	spamwriter = csv.writer(csvfile)
	for row in out:
		spamwriter.writerow(row)