
from __future__ import print_function
from labjack import ljm
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
import sys
import threading 
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
            for key, value in input.items()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    #elif isinstance(input, unicode):
    #    return input.encode('utf-8')
    else:
        return input

class LabjackInterface() :
    def __init__(self) :
        assert os.getuid() == 0, 'Must be run as root'
        # Open first found LabJack
        self.handle = ljm.openS("ANY", "ANY", "470018038")
		#Serial: 470018038
		#Name: IPSII_II_T7_8038
		#Eth. IP: 192.168.1.22
        print('here')
        self.t = threading.Thread(target=lambda : 0) # Dummy thread
        self.t.start()

        # Call eReadName to read the serial number from the LabJack.
        self.name = "SERIAL_NUMBER"
        result = ljm.eReadName(self.handle, self.name)
        print("\neReadName result: ")
        print("    %s = %f" % (self.name, result))


        # Load params
        with open("labjack_config.json") as cfg_file :
            cfg = byteify(json.load(cfg_file)) #turns the output of the json command and turns it into a dict
        self.scanTime = cfg['scanTime']
        self.scanRate = cfg['scanRate']
        self.scanRes = cfg['scanRes']
        self.settlingTime = cfg['settlingTime']
        self.chNames = cfg['chNames']
        self.chRange = cfg['chRange']

        assert len(self.chNames) == len(self.chRange), 'wrong number of ranges set'

        # Set parameters
        #self.scanTime = .6 # s
        #self.scanRate = 8000 # Hz
        #self.scanRes = 1 # Resolution index 0-8
        #self.settlingTime = 30
        #self.chNames = ['AIN'+str(i) for i in [0,1,2,3] ] # Pick channels to measure

        # Derived Parameters
        self.Nchannels = len(self.chNames)
        self.scanN = int(self.scanTime*self.scanRate)
        self.scanTimes= np.linspace(0, self.scanTime, self.scanN)


        self.scanparams = {
                "scanTime"      : self.scanTime,
                "scanRate"      : self.scanRate,
                "scanRes"       : self.scanRes,
                "settlingTime"  : self.settlingTime,
                "chNames"       : self.chNames,
                "chRange"       : self.chRange}

        # Prep chanel references
        self.channels = [ljm.nameToAddress(ch)[0] for ch in self.chNames]

        # Set device settings
        # Set range (gain)
        for i in range(self.Nchannels):
            ljm.eWriteName(self.handle, self.chNames[i]+'_Range',  self.chRange[i]) # Sets range (gain)

        # Stream resolution and settling
        ljm.eWriteName(self.handle, 'STREAM_RESOLUTION_INDEX', self.scanRes)   # Sets resolution (integration time)
        ljm.eWriteName(self.handle, 'STREAM_SETTLING_US', self.settlingTime)   # Sets settling time in us, max 4400

        ###
        ## Set up Triggered stream
        ###
        ljm.writeLibraryConfigStringS("LJM_STREAM_SCANS_RETURN","LJM_STREAM_SCANS_RETURN_ALL")
        ljm.writeLibraryConfigS("LJM_STREAM_RECEIVE_TIMEOUT_MS", 0)
        # 0 = no trigger, 2001 = trigger on DI01 (FI01)
        ljm.eWriteName(self.handle, "STREAM_TRIGGER_INDEX", 2001) 
        ljm.eWriteName(self.handle, "DIO1_EF_ENABLE", 0)
        ljm.eWriteName(self.handle, "DIO1_EF_INDEX", 4) #3 - rising edge, 4 - falling edge, 5- pulse
        ljm.eWriteName(self.handle, "DIO1_EF_CONFIG_A", 2)
        ljm.eWriteName(self.handle, "DIO1_EF_ENABLE", 1)

    def scan(self, saveFile,  debug=False, forceGoodData=False, syncToPi=True):
        '''
        Takes a single scan of 4 channels from the labjack and stores the result
        in 2 numpy files: currentdata/data.npy and [savefile]
        '''
        # Keep trying until you get some data
        successful = False

        # Start streaming
        scanRateAct = ljm.eStreamStart(self.handle, self.scanN, self.Nchannels, self.channels , self.scanRate)

        while not successful :
            # Try to take labjack data
            try :
                #t1 = time.time()
                # Triggger scan
                #ljm.eWriteName(self.handle, 'DAC0', 5.0)
                #ljm.eWriteName(self.handle, 'DAC0', 0.0)

                ## Read in data

                time.sleep(1e-3)
                rawdata = ljm.eStreamRead(self.handle)
                ljm.eStreamStop(self.handle)

                #streamBurst(handle, numAddresses, aScanList, scanRate, numScans)
                #aData=ljm.streamBurst(self.handle, self.Nchannels, self.channels, self.scanRate, self.scanN)
                #rawdata= [aData[1], aData[0], 0]
                #t2 = time.time()
                #print("Scan time: {}".format(t2-t1))


                #scanRateAct, rawdata =  ljm.streamBurst(self.handle, Nchannels, channels, scanN)
                if debug :
                    print('Data scan rate set, actual: {} Hz, {} Hz'.format(self.scanRate, scanRateAct))
                    print('Buffer size: {}, {}'.format(rawdata[1], rawdata[2]))
                successful = True
            except ljm.LJMError as e :
                print('Error: ')
                print(e)
                time.sleep(1);
                continue


            data = [ np.r_[rawdata[0][i::self.Nchannels]] for i in range(self.Nchannels)]

            # Check that data was good
            if forceGoodData :
                # Est. SNR of phase ref signal
                dstd = data[0].std()
                diffmean = np.abs(np.diff(data[0])).mean()
                snr_est = dstd/diffmean
                if snr_est < 3 :
                    print('\rBad data: estimated SNR = {}, time={}'.format(snr_est, time.ctime()), end='')
                    sys.stdout.flush()
                    time.sleep(1);
                    success = False
                    continue
        
        # Done taking data
        if debug :
            print('\n Done. Time={}'.format(time.ctime()))

        # Save data
        np.save(saveFile, data)
        #np.save('./currentdata/data.npy', data)
        if syncToPi : 
            self.t.join()
            self.t = threading.Thread(target=mysync, args=(saveFile,))
            self.t.start()
            # WARNING: FIX THIS!!!!!!! (amke sure the check that previous thread finished works)

        def __del__(self): #, exception_type, exception_value, traceback) :
            print("ljm closing") 
            ljm.halt(self.handle)

    def readValues(self, names):
        numFrames = len(names)
        return ljm.eReadNames(self.handle, numFrames, names)

def mysync(saveFile) :
	#rsync = "sudo -u coldatoms rsync -ra "+saveFile+" pi@ipsiipi:/home/pi/ipsii/currentdata/data.npy"
	rsync = "sudo -u ipsii_two rsync -ra "+saveFile+" carter@Carters_laptop:/home/carter/Documents/Research/Durfee/IPSII_II/currentdata/data.npy"
	os.system(rsync)

if __name__ == '__main__' :

    jack = LabjackInterface()
    print("Initialized")
    try :
        print('ctrl-c to quit')
        while True:
            jack.scan('testdata/data1.npy', debug=False, syncToPi=True)
            results = jack.readValues(["AIN4", "AIN6","AIN8"])
            print('\r Done. Time={}, '.format(time.ctime()) + 'Battery values: {}, {}, {}'.format(*results), end='')
            sys.stdout.flush()

    except KeyboardInterrupt:
        ljm.eStreamStop(jack.handle)
        1
