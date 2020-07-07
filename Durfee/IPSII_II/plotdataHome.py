import numpy as np
import sys
import os
import matplotlib
#tkagg -no good
#qt4agg - good
#wxagg -ok, but mem usage grows ~5%/hr

matplotlib.use('qt4agg',warn=False, force=True)
#matplotlib.use('wxagg',warn=False, force=True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import datetime
import threading as thread
#import myMessage
import json
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
            for key, value in input.items()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
#    elif isinstance(input, unicode):
#        return input.encode('utf-8')
    else:
        return input

#plt.ion()
#plt.style.use('seaborn-dark') # customize your plots style

# Script args
messageInt = 10
messageSkip = messageInt
helpstring = '''
python plothomeData datafile [options]
-m : Display a flashing message warning people to be cautious
-h : show help
'''
if '-h' in sys.argv :
    print(helpstring)
    sys.exit()
global messageDisplay
messageDisplay = False
if '-m' in sys.argv :
    messageDisplay = True
dataFile = sys.argv[1]

# Parameters
# Needs to be updates to use actual data
# !!!!!!!!!!!!
#scanTime = .6 # s
#scanRate = 8000 # Hz
#scanRes = 1 # Resolution index 0-8
#settlingTime = 30

# Loaded from actual data (?)
#os.system("rsync -ra coldatoms@ipsiibb:/home/coldatoms/ipsii/labjack_config.json /home/pi/ipsii/")
with open("labjack_config.json") as cfg_file :
        cfg = byteify(json.load(cfg_file))
        scanTime = cfg['scanTime']
        scanRate = cfg['scanRate']
        scanRes = cfg['scanRes']
        settlingTime = cfg['settlingTime']
        chNames = cfg['chNames']
        chRange = cfg['chRange']
Nchannels = len(chNames)
chRange = [10]*Nchannels
scanN = int(scanTime*scanRate)# must equal 500
scanTimes= np.linspace(0, scanTime, scanN)
#print(scanN)
chNames = ['AIN'+str(i) for i in [0,1,2,3] ] # Pick channels to measure

# Setup plot
fig = plt.figure(1, figsize = (10,8), dpi=100)
plt.clf()
pltsty = ['r', 'g', 'b', 'y']
plt.subplot(Nchannels+1, 1, 1)
myslice = slice(0, int(7*len(scanTimes)/10) )
lines1 = [plt.plot(scanTimes[myslice], scanTimes[myslice]*0, pltsty[i], label='{}'.format(chNames[i], 1.0/chRange[i]))[0] for i in range(Nchannels)]
legend = plt.legend(loc = 'upper right', title='         mean          std          snr')
legendText =  legend.get_texts()
plt.ylim([-10, 10])
plt.xlim([0, scanTime])
lines2=[]
for i in range(Nchannels) :
    plt.subplot(Nchannels+1,1,2+i)
    lines2 += [ plt.plot(scanTimes, scanTimes*0, pltsty[i], label='{}'.format(chNames[i], 1.0/chRange[i]))[0] ]
    legendText[i].set_text('{}: {:05.2f},  {:07.2e},  {:04f}'.format(chNames[i], .0001, .0001, .0001))
    plt.xlim([0, scanTime])
    plt.ylim([-10, 10])
plt.tight_layout()
#plt.show(False)
#plt.draw()
#plt.pause(.1)

# Continually plot until interrupt
def animate(ii):
    global messageDisplay, messageSkip, messageInt, dataFile
    fileRead = False
    dataShape = (0,0)
    #dataFile = '/home/pi/ipsii/currentdata/data.npy'
    #dataFile = 'data072/step_x_0001_y_0001.npy'
    while not fileRead:
        try :
            # Get data
            time.sleep(0.01)
            #os.system("rsync -ra coldatoms@ipsiibb:/home/coldatoms/ipsii/currentdata /home/pi/ipsii/")
            time.sleep(0.01)
            data = np.load(dataFile)
            #print("before if/elif", "\t\t\t", "dataShape = %s" % str(dataShape), "\t\t\t", "data.shape = %s" % str(data.shape))
            time.sleep(0.01)
            if dataShape == (0,0) : # First time save the shape of the file for future
                #print("if statement", "\t\t\t", "dataShape = %s" % str(dataShape), "\t\t\t", "data.shape = %s" % str(data.shape))
                dataShape = data.shape
            elif dataShape == data.shape :
                fileRead = True
                #print("elif statement", "\t\t\t", "dataShape = %s" % str(dataShape), "\t\t\t", "data.shape = %s" % str(data.shape))
            else :
                print("Wrong shape. Read interupted?")
                continue

            # Display message
            #filet = os.path.getmtime('/home/pi/ipsii/currentdata/data.npy')
            filet = 0
            proct = time.time()
            #print('Age of current data: ' + str(datetime.timedelta(seconds=proct-filet)))
            #print(proct-filet)
            messageSkip += 1
            #if proct-filet < 1000 and messageDisplay  and messageSkip >= messageInt:
                #myMessage.display_text("Scan in progress\nDon't bump, breathe on, or look at table\n",1)
                #myMessage.display_text("Scan in progress - Don't touch table, \n unless you're working on something \n in which case, just be careful",1)
                #messageSkip = 0

        except (IOError, ValueError) as e :
            print(e)
            print("Plotting Continuing")
            continue
            time.sleep(.3)

        #print("before for statement", "\t\t\t", "dataShape = %s" % str(dataShape), "\t\t\t", "data.shape = %s" % str(data.shape))
        for i in range(Nchannels):
            dmean = data[i].mean()           # signal average
            dstd = data[i].std()             # deviation
            diffmean = np.abs(np.diff(data[i])).mean() # diff deviation
            #snr_est = dstd/diffmean   # snr estimate
            snr_est = data[i][int(len(data[i])/6):].std()/data[i][0:int(len(data[i])/6)].std()
            lines1[i].set_ydata(data[i][myslice])
            legendText[i].set_text('{}: {:05.2f},  {:07.2e},  {:03.0f}'.format(chNames[i], dmean, dstd, snr_est))
            #print("in for statement", "\t\t\t", "dataShape = %s" % str(dataShape), "\t\t\t", "data.shape = %s" % str(data.shape))
            # Scaled
            #lines2[i].set_xdata( scanTimes )
            lines2[i].set_ydata( (data[i]-dmean)/dstd*2)

            #legend.remove()
            #legend = plt.legend()
            #print(lines1+lines2+[legend])
    return lines1+lines2+[legend]
while True :
    try :
        #print("while True:\n\ttry:\n\t\tani = animation.FuncAnimation...")
        ani = animation.FuncAnimation(fig, animate, interval=200, blit=True, frames = 5, repeat=True)
        #print("before plt.show()")
        plt.show()
        #print("after plt.show()")
    except  RuntimeError :
        print('Error')



