

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

samples = 2000
f1 = 25 			# signal frequency
Fs = 1000		# sample frequency
x = np.arange(samples)
sin1 = np.sin(2 * np.pi * f1 * x / Fs) * 2
sin2 = np.sin(2 * np.pi * 2 * x / 200) * x / 100
sin3 = np.sin(2 * np.pi * 0.5 * x / 1000) * 3
lin1 = signal.sawtooth(2 * np.pi * 5 * x / 800) * 3
lin2 = signal.sawtooth(2 * np.pi * 5 * x / 2000) * 20
tri1 = signal.triang(samples) * 4
sq1 = signal.square(2 * np.pi * 30 * x / 1000) * 5
# y = sin1 + sin2  + lin1 + lin2
# y = lin1 + sq1
y = sin1 + sin2
plt.plot(x, y)
plt.show()


f = open("mydata2.csv", "w")
f.write( str("Close\n"))
for i in range(samples):
	f.write(str(y[i]) + "\n")

