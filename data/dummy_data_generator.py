

import matplotlib.pyplot as plt
import numpy as np

samples = 5000
f1 = 25 			# signal frequency
Fs = 1000		# sample frequency
x = np.arange(samples)
sin1 = np.sin(2 * np.pi * f1 * x / Fs) * 20
sin2 = np.sin(2 * np.pi * 2 * x / 200) * x / 100
sin3 = np.sin(2 * np.pi * 0.5 * x / 1000) * 20 / x
y = sin1 + sin2 + sin3
plt.plot(x, y)
plt.show()


f = open("mydata1.csv", "w")
f.write( str("Close\n"))
for i in range(samples):
	f.write(str(y[i]) + "\n")

