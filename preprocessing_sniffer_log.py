"""pre-processing sniffer log, to make each consecutive record has 1/30s time different."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, help='path to log-file', default='')
args = parser.parse_args()
path_to_log = args.p

time_interval = 33

##load record log into a list
data = list()
new_entries = ''
with open(path_to_log+'log.txt','r') as input_file:
	for line in input_file:
		data.append(line.split(','))

REMAIN = False
STARTED = False

for entry in data:

	time = int(entry[0][10:])
	key = entry[5].replace('\n','')
	key = key.replace('\r','')
	
	if not STARTED:
		
		new_entries += '{0},{1},{2},{3},{4},{5}\n'.format(entry[1],entry[2],entry[3],entry[4],key,time)
		STARTED = True
		last_time = time

	else:

		time_diff = time - last_time

		if time_diff > time_interval-1:
			if time_diff < time_interval+1:

				new_entries += '{0},{1},{2},{3},{4},{5}\n'.format(entry[1],entry[2],entry[3],entry[4],key,time)
				last_time = time
				REMAIN = False

			else:

				NO_rec = time_diff/time_interval
				rem = time_diff%time_interval

				if REMAIN:

					new_entries += '{0},{1},{2},{3},{4},{5}\n'.format(fname,x_loc,y_loc,mouse_button,keyboard_input,time)
					for i in range(1,NO_rec):
						new_entries += '{0},{1},{2},{3},{4},{5}\n'.format(entry[1],entry[2],entry[3],entry[4],key,time)

				else:
					
					for i in range(NO_rec):
						new_entries += '{0},{1},{2},{3},{4},{5}\n'.format(entry[1],entry[2],entry[3],entry[4],key,time)

				last_time += time_interval*NO_rec

				if rem > 0:

					fname = entry[1]
					x_loc = int(entry[2])
					y_loc = int(entry[3])
					mouse_button = entry[4]
					keyboard_input = key
					REMAIN = True

				else:
					REMAIN = False

		else:

			fname = entry[1]
			x_loc = int(entry[2])
			y_loc = int(entry[3])
			mouse_button = entry[4]
			keyboard_input = key
			REMAIN = True

with open(path_to_log+'log_processed.txt','w') as f:
	f.write(new_entries)