import numpy as np
import sys
import json
import os

folder = sys.argv[1]
params = json.load(open('dataruns/'+folder+'params_set.json'))
N = params['x_lastN']
zfill_value = len('{}'.format(N)) + 1
#print(zfill_value)
pixels_1D = 2*N+1
num_channels = 4
size_of_good_data = 500
bigdata = np.empty((pixels_1D, pixels_1D, num_channels, size_of_good_data))
newfile = 'all_the_data'

#print(folder, np.shape(bigdata), str(-3).zfill(3))
def update_progress(workdone):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

i = 0
for x in range(-N,N+1):
    for y in range(-N,N+1):
        data = np.load('dataruns/'+folder+'step_x_'+str(x).zfill(zfill_value)+'_y_'+str(y).zfill(zfill_value)+'.npy')
        bigdata[x,y,:,:] = data

#        os.remove(folder+'step_x_'+str(x).zfill(zfill_value)+'_y_'+str(y).zfill(zfill_value)+'.npy')

        i += 1
        update_progress(i/(pixels_1D**2))

print()
np.save(folder+newfile,bigdata)


### The resulting data file is in the format data(x,y,channel,data)
