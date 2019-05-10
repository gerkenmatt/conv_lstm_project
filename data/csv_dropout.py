
import sys

data_file = sys.argv[1]
print("file name: ", str(sys.argv[1]))

new_file_name = data_file.split('.')[0] + "_dropout.csv"
print("new file name: ", new_file_name)

dropout = 4	# number of lines to save before we remove a line
drop = 0

with open(data_file, 'r') as f: 
	with open(new_file_name, 'w') as g:
		for x in f: 
			if drop == 0:
				g.write(x)
				drop += 1
			elif drop  >= dropout: 
				drop = 0
			else:
				drop += 1
