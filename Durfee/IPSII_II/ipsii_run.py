from __future__ import print_function
import numpy as np
import os
import time
import json
import sys
import traceback

# My stuff
import interferometer
import mylabjack


def setupFolder(rootFolder, resume=False):
    '''
    Load/Save folder
    if continuing pick last data folder,
    otherwise create new folder

    Folders seq numbered 'data001', 'data002', etc.
    '''
    i = 0
    while True:
        i += 1
        dataFolder = rootFolder + 'data{:03n}/'.format(i)
        if not os.path.exists(dataFolder):
            break
        dataFolder_prev = dataFolder
    if resume:
        dataFolder = dataFolder_prev
        print('Resuming prev data run. Missing data will be filled in')
        assert False,'Not supported yet'
    else:
        os.makedirs(dataFolder)
    return dataFolder


def takeData(setparams, jack, intf, dataFolder, status=True, returnHome=True):
    '''
    setparams should include:
        x_firstN - start step x direction
        x_lastN - last step x direction
        y_firstN - start step x direction
        y_lastN - last step x direction
        x_stepN - number of steps to skip in kspace
        y_stepN - number of steps to skip in kspace
        x_FOV  - x field of view in mm
        y_FOV  - y field of view in mm

    jack - labjack interface object
    intf - interferometer object
    '''
    s = setparams
    d = deriveParams(s)
    dkx = d['x_kres']  # should be lines/mm
    dky = d['y_kres']
    Ntotal = d['Npoints']

    # Fields width for steps - eg. # decimal points + 1 for minus sign
    Nfield = str(int(np.log10(max(abs(s['x_firstN']), abs(s['y_firstN']), s['x_lastN'], s['y_lastN']) )) + 2)

    # Go to starting point for x and y, with backlash correction
    intf.kSet(kx=s['x_firstN']*dkx, ky=s['y_firstN']*dky)
    intf.backlashPositive()

    # Keep track of how many points
    ii = 1

    t1 = time.time()
    # Scan over kx and ky
    try:
        for ys in range(s['y_firstN'], s['y_lastN']+s['y_stepN'], s['y_stepN']):
            # Check battery values
            battery_levels= jack.readValues(["AIN4", "AIN6","AIN8"])
            with open(dataFolder+'a_log_file.txt', 'a') as logf : 
                update_string = 'Row={}, Time={}, '.format(ys, time.ctime())+' Battery values: {}, {}, {}\n'.format(*battery_levels)
                print(update_string)
                logf.write(update_string)

            # Go to starting point for x, with backlash correction
            intf.kSet(kx=s['x_firstN']*dkx)
            intf.backlashPositive()
            for xs in range(s['x_firstN'], s['x_lastN']+s['x_stepN'], s['x_stepN']):
                # Create file name for current step
                stepstring = ('step_x_{0:0'+Nfield+'}_y_{1:0'+Nfield+'}').format(xs,ys)
                saveFile = dataFolder+stepstring+'.npy'

                # Update status
                if status:
                    t2 = time.time()
                    avgT = 1.0*(t2-t1)/ii
                    remainingT = (Ntotal -ii)*avgT
                    print(('{0:0'+Nfield+'}' +
                        '/{1:0'+Nfield+ '},' +
                        ' est time remaining: {2} min {3} s , saving: '
                        ).format(ii, Ntotal, int(remainingT/60), int(remainingT%60)) + saveFile, end='        \r')
                    sys.stdout.flush()

                # Check if file exists
                # ! not implemented !

                # Make sure we're done moving
                for t in interferometer.StepperMotor.runningMotors.values() : t.join()
                time.sleep(.01)

                # Take data
                jack.scan(saveFile, debug=False, forceGoodData=False)

                # Step kx
                intf.kAdd(dkx, 0)

                # Update counter
                ii+=1

            # Step ky
            intf.kAdd(0, dky)
        print('\nDone')
    except KeyboardInterrupt:
        print('Process Interrupted')

    t2 = time.time()
    totalT = t2-t1
    print('Total time: {} min {} s , saving: '.format(int(totalT/60), totalT%60))
    print('Interferometer angle: {}'.format(intf.angle))
    if returnHome:
        print('Returning home')
        intf.kSet(0.0,0.0)
        intf.backlashPositive()

    return totalT


def deriveParams(sparams):
    '''
    takes set params dictionary and
    calculates some other useful stuff
    '''
    dparams = {} #collections.defaultdict(dict)
    dparams['x_N'] = int((sparams['x_lastN']-sparams['x_firstN'])/sparams['x_stepN'])+1
    dparams['y_N'] = int((sparams['y_lastN']-sparams['y_firstN'])/sparams['x_stepN'])+1
    dparams['Npoints'] = dparams['x_N']*dparams['y_N']

    dparams['x_kres'] = 1.0/sparams['x_FOV'] # in mm^-1
    dparams['y_kres'] = 1.0/sparams['y_FOV'] # in mm^-1

    dparams['x_kmax'] = dparams['x_kres']*max(abs(sparams['x_firstN']), sparams['x_lastN'])
    dparams['y_kmax'] = dparams['y_kres']*max(abs(sparams['y_firstN']), sparams['y_lastN'])

    if not dparams['x_kmax'] == 0:
        dparams['x_res'] = 1.0/(2*dparams['x_kmax'])
    else:
        dparams['x_res'] = 0
    if not dparams['y_kmax'] == 0:
        dparams['y_res'] = 1.0/(2*dparams['y_kmax'])
    else:
        dparams['y_res'] = 0
    return dparams



# Check that I'm running in a screen session so process continues if ssh disconnected
if 'STY' in os.environ and os.environ['STY'] != '' :
    print('screen session detected')
else :
    print("WARNING: Not running in screen (can't resume if disconnected)")
    raw_input('press key to continue')

# Initialize labjack
jack = mylabjack.LabjackInterface()

# Initialize interferometer
intf = interferometer.Interferometer()

# Set up folder
resume = sys.argv[0] == 'cont'
#rootFolder = '/mnt/coldatoms/ipsii/'
#rootFolder = '/mnt/piDrive/dataRuns/'
rootFolder = '/media/ipsii_ii_drive/dataruns/'
dataFolder = setupFolder(rootFolder, resume)

# Setup parameter dict
# 40x40 pixels takes 45min
params_set = {} #collections.defaultdict(dict)
L = 150
params_set['x_firstN'] = -L #150 
params_set['x_lastN']  =  L
params_set['y_firstN'] = -L
params_set['y_lastN']  =  L
params_set['x_stepN'] = 1
params_set['y_stepN'] = 1
params_set['x_FOV'] = 4.0 # mm
params_set['y_FOV'] = 4.0 # mm
params_set['threadingOn'] =  True
params_set['img_x_shift']  = 0
params_set['img_y_shift']  = 0

# Single arm mode?
params_set['single_arm']  = 1
intf.setSingleArm(params_set['single_arm'])

# Calculate additional paramters
params_derived = deriveParams(params_set)


# Zero interferometer
print()
sys.stdout.flush()
interferometer.manualMove(intf, incr = .0001, message=" Please zero interferometer, coming from negative on both axis (pressing d and l or w and i)")
intf.resetHome()

# Test data run
jack.scan('testdata/data1.npy', debug=False)
print("Check that trial data run worked all right")
print("Check that beams are unblocked, and offset is set correctly)")

# Get additional notes
params_set['notes'] = ""
params_set['notes'] += raw_input("Enter any additional notes about datarun:")

# Set up params, and save
with open(dataFolder+"params_set.json", 'w') as f: json.dump(params_set, f, sort_keys=True, indent=4)
with open(dataFolder+"params_derived.json", 'w') as f: json.dump(params_derived, f, sort_keys=True, indent=4)
with open(dataFolder+"params_labjack.json", 'w') as f: json.dump(jack.scanparams, f, sort_keys=True, indent=4)

# Print a few things
print( 'FOV = ({},{}) mm'.format(params_set['x_FOV'],     params_set['y_FOV']) )
print( 'res = ({},{}) mm'.format(params_derived['x_res'], params_derived['y_res']) )

# Disable threading
if not params_set['threadingOn'] :
    intf.threadingOff()

# Take data
try :
    totalT = takeData(params_set, jack, intf, dataFolder)
    params_derived['time'] = totalT
    with open(dataFolder+"params_derived.json", 'w') as f: json.dump(params_derived, f, sort_keys=True, indent=4)
except Exception as e :
    traceback.print_exc()
    print(e)
    print("\nWarning, exception occurred")
    print("YOU need to MANUALLY return home if desired")
    print('i.e.: intf.kSet(0.0,0.0); intf.backlashPositive()')
    intf.turnOff()
#
intf.threadingOn()

# End
for t in interferometer.StepperMotor.runningMotors.values() : t.join()
intf.turnOff()

from IPython import embed
embed()
