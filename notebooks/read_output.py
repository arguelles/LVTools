import struct
import numpy as np

shape = (1,100)

def read_bin(filename):
	with open(filename, 'rb') as file:
		data = struct.unpack('@' + str(np.prod(shape)) + 'd', file.read())
		array = np.array(data).reshape(shape)
		return array

if __name__ == '__main__':
	import sys
	array = read_bin(sys.argv[1])

	import matplotlib.pyplot as plt
	plt.pcolor(array, cmap=plt.cm.Blues)
	plt.colorbar()
	plt.show()