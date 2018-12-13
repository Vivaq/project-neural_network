import numpy

data = numpy.load('data.npy')

min_max = []
for arr in data:
	label = arr[0]
	features = arr[1:]

	min_max.append((features.min(), features.max()))

for i in min_max:
	print(i)

