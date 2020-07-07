import numpy as np
import matplotlib.pyplot as plt
import sys
import json

folder = sys.argv[1]
params = json.load(open(folder+'params_set.json'))
N = params['x_lastN']
all_the_data = np.load(folder+'all_the_data.npy')

ref = all_the_data[:,:,0,:]
plt.plot(ref[int(N/2),int(N/2),:])
plt.show()


