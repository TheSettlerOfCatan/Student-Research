from __future__ import print_function
from six.moves import zip
from six import print_
print = print_
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize
import matplotlib as mpl
mpl.use('qt4agg')
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import os
import sys
import json
import time
import multiprocessing as multi #import Pool, Queue, Process, Lock
#import Image
import traceback
import hsluv # install via pip

if sys.version_info.major > 2 :
    myinput = input
else :
    myinput = input

#if sys.version_info.major > 2 :
#    izip = zip
#else :
#    from itertools import izip
#from numba import jit
#sys.settrace()
mpl.rcParams['image.cmap'] = 'bone'
#plt.ion()

def backProjection(im, d, k, r=None, c=None, off=None, angle=None, skip=1) :
    '''
    Attempt to back project from the image plane using a brute force
    Huygens principle approach with a 2D complex image
    d = distance to back project (in pixels)
    k = wavenumber (in 1/pixels)
    r = L1 radius of area to calculate
    c = (row,column) center of area to calculate
    off = pixel offset
    angle = (small) angle to back project to (i.e. recenters image)
            replaces 'off'
    '''
    if c==None:
        c = (im.shape[0]/2,im.shape[1]/2)
    if off==None:
        off = (0,0)
    if angle != None :
        off = ( int(d*np.sin(angle[0])), int(d*np.sin(angle[1])) )
        print('Offset for back projection in pixels: {}'.format(off))

    if r is not None :
        row_start  = c[0]-r
        col_start  = c[1]-r
        row_finish = c[0]+r
        col_finish = c[1]+r
    else :
        row_start  = 0
        col_start  = 0
        row_finish = im.shape[0]
        col_finish = im.shape[1]
    # Huygens calculation
    # each spot G(i,j) = sum im(l,m) point sources
    # point source: A*e^i (k*R_ijlm+phase) where R is the distance from the old point
    # to the new
    X, Y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    X += off[1]
    Y += off[0]
    A = np.abs(im)
    P = np.angle(im)
    im2 = im*0
    tstart = time.time()
    tp = 10**10
    for j in range(row_start,row_finish, skip) :
        for i in range(col_start,col_finish, skip) :
            R2 = d**2+(j-Y)**2+(i-X)**2
            R = np.sqrt(R2)
            im2[j,i] = np.mean(A/R*np.exp(-1j*(k*R-P)))
        tdel = time.time() - tstart
        tleft = (im.shape[0]-j+row_start) * tdel/(j-row_start+1)
        if tp-tleft > 10 :
            tp = tleft
            print('Row {}/{}, est time remaining: {}m {}s'.format(j,im.shape[0], int(tleft/60), int(tleft) % 60 ))
    return im2

# Make nice colormap for cyclic phase images
# Courtesy of https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots
def make_anglemap( N = 256, use_hpl = True ):
        h = np.ones(N) # hue
        h[:N//2] = 11.6 # red
        h[N//2:] = 258.6 # blue
        s = 100 # saturation
        l = np.linspace(0, 100, N//2) # luminosity
        l = np.hstack( (l,l[::-1] ) )

        colorlist = np.zeros((N,3))
        for ii in range(N):
            if use_hpl:
                colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s, l[ii]) )
            else:
                colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s, l[ii]) )
        colorlist[colorlist > 1] = 1 # correct numeric errors
        colorlist[colorlist < 0] = 0
        return pltcolors.ListedColormap( colorlist )


def myunwrap(a, m=None):
    ''' partially 'unwraps' phase data for better
    looking plots
    TESTING - not done
    m = mask of places to ignore
    '''
    if m == None :
        m = 0*a+1

    #return a%(np.pi) # wrap phase to (0,2pi) range
    return np.arctan(np.tan(a))
    #a1 = a+np.pi
    #a2 = a-np.pi
    #b = np.roll(a, (1,1))
    #cond1 =  (a1-b)**2 < (a-b)**2
    #cond1[::2,::2] = False
    #a[cond1] = a1[cond1]
    #cond2 =  (a2-b)**2 < (a-b)**2
    #cond2[::2,::2] = False
    #a[cond2] = a2[cond2]
    #return a


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

def plotFit(f, x, data, p0, label='', newfig=True) :
    '''
    plots the function
    f - function model
    x - the points at which to plot
    p0 - paramters of the model
    data - the actual y data
    label - label to display on plot
    '''
    if newfig :
        plt.figure(2, figsize=(15,5), dpi=150)
        plt.clf()
    fit = f(x, *p0)
    plt.plot(x, data, '.', label=label)
    plt.plot(x, fit, 'r', label='Fit: '+f.func_name, lw=2)
    plt.legend()
    plt.show(False)
    plt.pause(.1)
    #input()
    #plt.close()

def rampModel(ix, y_min, y_max, i0, T, sym, cycStart, cycStop) :
    '''
    return y = ramp(x)
    ix = x index series (0, 1, 2, ...)
    y_min - min
    y_max - max
    i0    - x offset of ramp signal (basically the 'phase' of the ramp)
    T     - period for ramp
    sym   - symmetry of ramp (eg. traiangle=.5, sawtooth=1, reverse sawtooth=0)
    cycstart - x location where ramp signal turns on
    cycstart - x location where ramp turns off
    '''
    # remove this if you want symmetry to be a fit param
    #sym=.5
    ioff = (ix-i0)*1.0
    isub = (ioff%T)*1.0

    upT = sym*T
    downT = (1-sym)*T

    up = 1.0*isub/upT
    down = 1.0*(T-isub)/downT

    iupMask = (isub<T*sym).astype(int)
    idownMask = (isub>=T*sym).astype(int)

    y = up*iupMask + down * idownMask

    y[ioff < cycStart*T] = y[np.argmin( (ioff - cycStart *T)**2 ) ]
    y[ioff > cycStop*T] =  y[np.argmin( (ioff - cycStop  *T)**2 ) ]

    y = y*(y_max-y_min) +y_min
    return y

#@jit
def ramp_gauss(ix, *p0) :
    return p0[0] + (p0[1]-p0[0])*np.exp(-(ix-p0[2])**2/p0[3]**2*4)
#bad @jit
def errfRAMP1(p0, ix, ramp_data) :
    return np.sum((ramp_gauss(ix, *p0) - ramp_data)**2)
#bad @jit
def errfRAMP2(p, ix, ramp_data) :
    return np.sum((rampModel(ix, *p) - ramp_data)**2)

#@jit
def rampFit(ix, ramp_data, Ncyc=1, symest=.8, plot=False) :
    '''
    Fits a ramp of the type defined in rampModel
    ix - x axis (integer series)
    ramp_data - data
    Ncyc - number of times the ramp repeats
    symest - estimate of the ramp symmetry (will be fit later)
    '''
    # Guess some values from the given data
    ramp_min = ramp_data.min()
    ramp_max = ramp_data.max()
    ramp_range = ramp_max-ramp_min
    ##dRampi = (ramp_data*30/ramp_range).astype(int) #Integer version
    ##Test = (np.argmax(ramp_data)-np.argmin(ramp_data))/symest
    #Test = len(ix)/2
    #cycstartest = (ramp_data[0] - ramp_min)/ramp_range * (symest-1)

    ## Try gauss fit first to see where (first?) bump is
    #peak_est = np.argmax(ramp_data)
    #p0 = [ramp_min, ramp_max, peak_est, Test, symest, cycstartest, cycstartest+Ncyc]
    #sol = scipy.optimize.minimize(errfRAMP1, p0, args=(ix, ramp_data), method='Powell', options = {'maxfev':5000}, tol=1e-5)
    #popt = sol['x']

    ## Now fit actual ramp
    #Test = popt[3] *  np.pi/2
    #p0 = [popt[0], popt[1], popt[2]-Test/2, Test, symest, 0, 1 ]

    peak_est = np.argmax(ramp_data)
    Test = len(ix)*.9
    cycstartest=0
    p0 = [ramp_min, ramp_max, peak_est-Test*symest, Test, symest, cycstartest, cycstartest+Ncyc]

    sol = scipy.optimize.minimize(errfRAMP2, p0, args=(ix, ramp_data), method='Powell', options = {'maxfev':5000}, tol=1e-5)
    success =  np.sqrt(sol['fun'])/len(ix) < ramp_range**2/20
    popt = sol['x']
    if plot :
        print(sol)
        plotFit(rampModel, ix, ramp_data, p0, 'Ramp')
        input("Press key to continue")
        plotFit(rampModel, ix, ramp_data, popt, 'Ramp')
        input("Press key to continue")

    return popt, p0, success

#@jit
def phaseModel(ix, *p) :
    '''
    simple polynomial
    '''
    return np.polyval(p,ix)

def phaseFit(x,y, plot=False) :
    '''
    Estimate a model for a non linear ramp
    i.e. find the instantaneous phase of the ref signal
    '''
    phase = np.unwrap(np.angle(sig.hilbert(y-y.mean())))
    popt = np.polyfit(x,phase,5)
    pline = np.polyfit(x,phase,1)
    if plot:
        print(popt)
        fphase = phase-phaseModel(x,*pline)
        pnonlinear = popt.copy();
        pnonlinear[-2:] -= pline
        plotFit(phaseModel, x, fphase, pnonlinear, 'InstaPhase nonlinearity')
        plt.plot(x, (y-y.mean())/(y.max()-y.mean()))
        plt.ylabel('Radians')
        plt.show(False)
        plt.pause(.1)
        input("Press key to continue")
    return phaseModel(x, *popt) # OR just return phase?? (i.e. instant phase - not model)

def resampleData(x, data, oversample=1, plot=False) :
    '''
    Resamples the data to be at regular intervals
    oversample - factor to increases the number of points by
    '''
    x_new = np.linspace(x.min(), x.max(), int(data.shape[1]*oversample))
    data_new = np.array([ interp.griddata(x, d, x_new) for d in data] )

    if plot :
        plt.clf()
        plt.plot( data[0], label='uncorrected')
        plt.plot(data_new[0], label='corrected')
        plt.legend()
        #plt.xlim([800,1000])
        plt.draw(); plt.pause(.1)
        input('Press key to continue')
    return x_new, data_new

def deTrend(x,y) :
    b = np.polyfit(x,y,1)
    y2 = y - b[0]*x-b[1]
    return y2

def sinModel(x, y0, A, freq, phase) :
    return y0+A*np.sin(2*np.pi*freq*x + phase)

#@jit #- bad
def errfSIN(p0, x, y) :
    return np.sum(( sinModel(x, *p0) - y)**2)

#@jit
def sinFit2(x, yin, thresh, plot=False) :
    ''' Threshold < period but > jitter in N data points'''

    y = yin
    success=True
    # Fit to sin wave
    # Adapted from
    # https://www.mathworks.com/matlabcentral/answers/121579-curve-fitting-to-a-sinusoidal-function
    thresh = int(thresh)
    yu = max(y)
    yl = min(y)
    yr = (yu-yl)                                # Range of 'y'
    yz = y-yu+(yr/2)
    #zx = x[yz *np.roll(yz,1) <= 0]             # Find zero-crossings
    zx = x[1:][yz[0:-1]*yz[1:] <= 0]                # Find zero-crossings
    pers = np.diff(zx)                          # calculate distances between crossings
    dx = np.median(np.diff(x))
    perThreshold = thresh*dx                    # value guaranteed to be < period (but > then jitter effect)
    pers = pers[pers>perThreshold]              # remove bad values from jitter
    if len(pers) < 2 :
        success = False
        print('Warning, period estimate failed. Adjust thresh')
        per=10
    else :
        per = np.percentile(pers, 50)*2.1             # Estimate period
    ym = (yl+yu)/2                              # Estimate offset

    i0 = np.argmin(np.abs(x))
    phase = np.arcsin((y[i0]-ym)/yr)             # Estimate phase
    q = np.diff(y)[min(i0-thresh, 0):max(i0+thresh, len(y))]
    #1/len(q)
    if not len(q) > 0:
        print("sin fit couldn't estimate phase")
        success=False
    if  q.mean() < 0:
        phase+=np.pi

    p0 = ( ym, yr/2, 1.0/per, phase)

    # Least squares fit
    sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)

    # Check sol, and try again if necessary
    errMax = np.sum((y-y.mean())**2) # If fit were just a line
    p0_first = p0
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, 1.1, 1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, .9, 1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, 1, -1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        for ri in range(30) :
            rnums = .1*(1+ri//6)*(np.random.rand(2)*2+1)
            p0 = p0_first*np.array([ 1, 1, 1+rnums[0], 1+rnums[1]])
            sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Nelder-Mead', options = {'maxfev':4000}, tol=1e-8)
            if sol['fun'] < errMax/4 :
                if (ri > 3):
                    print('Monte-carlo approach - found it on try {}   '.format(ri))
                break
    if sol['fun'] > errMax/4 :
        success=False
        print("sin fit appears to have failed")


    popt = sol['x']

    # Correct 180 phase shift
    if popt[1] < 0 :
        popt[1] *= -1;
        popt[3] += np.pi

    if plot :
        plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.title("sinFit2")
        plt.plot(x,y)
        plt.hlines([0,thresh], x.min(), x.max()/5)
        #print(sol)
        #plotFit(sinModel, x, y, p0, 'Slit guess')
        #plt.vlines(x[i0], y.min(), y.max())
        #input("HI")
        plt.subplot(2,1,2)
        plotFit(sinModel, x, y, popt, 'Slit fit', newfig=False)
        input("Press key to continue")

    if not sol['success'] :
        return [0,0,0,0], False
    return popt, success


#sinFitHybrid
def sinFitHybrid(x,y, thresh, plot=False) :
    dx = np.diff(x).mean()
    N = len(x)

    # Fourier transform
    yF = np.fft.fft(y, norm='ortho')

    # Find peak freq - ignore DC component
    imax = np.argmax(np.abs(yF[1:N//2]))+1

    # Calculate amplitude, etc. from value of peak
    f_fit = 1.0/dx*imax/N
    A_fit = np.abs(yF[imax])/4
    phi_fit = np.angle(yF[imax])+ np.pi/2
    y0_fit = y.mean()
    p0 = (y0_fit, A_fit, f_fit, phi_fit)

    # Least squares fit
    errfSIN = lambda p0, x, y : np.sum(( sinModel(x, *p0) - y)**2)
    sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)

    # Monte carlo phase search
    success = True
    errMax = np.sum((y-y.mean())**2) # If fit were just a line
    p0_first = p0
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, 1.1, 1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, .9, 1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        p0 = p0_first*np.array([ 1, 1, 1, -1])
        sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Powell', options = {'maxfev':4000}, tol=1e-8)
    if sol['fun'] > errMax/4 :
        print('Monte carlo search')
        for ri in range(30) :
            rnums = .1*(1+ri//6)*(np.random.rand(2)*2+1)
            p0 = p0_first*np.array([ 1, 1, 1+rnums[0], 1+rnums[1]])
            sol = scipy.optimize.minimize(errfSIN, p0, args=(x,y), method='Nelder-Mead', options = {'maxfev':4000}, tol=1e-8)
            if sol['fun'] < errMax/4 :
                if (ri > 3):
                    print('Monte-carlo approach - found it on try {}   '.format(ri))
                break
    if sol['fun'] > errMax/4 :
        success=False
        print("sin fit appears to have failed")

    y0_fit, A_fit, f_fit, phi_fit = sol['x']

    # Clean up results
    if A_fit<0 :
        A_fit*=-1
        phi_fit+=np.pi
    phi_fit = phi_fit%(2*np.pi) # wrap phase to (0,2pi) range
    popt = [y0_fit, A_fit, f_fit, phi_fit]

    if plot :
        plt.figure(2)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(np.abs(yF), '.', label='fft')
        plt.vlines(imax,np.abs(yF.min()), np.abs(yF.max()))
        plt.subplot(2,1,2)
        plt.plot(x,y, '.', label='data')
        plt.plot(x,sinModel(x, *popt), '-', label='fit')
        plt.show(False)
        plt.pause(.1)
        input('Press key to continue')
    return popt, success



def noiseEstimate(y, i=0) :
    if len(y) < 3 :
        return 0
    else :
        lv = int(np.log2(len(y)))
        lv=2.0
        return ( 1.0/lv*np.mean(np.abs((y[1:-1]-(y[:-2]+y[2:])/2)))
                + (lv-1.0)/lv*noiseEstimate(y[::2], i=i+1) )

def snrEstimate(y):
    '''
    Assumes data spacing is small compared to the signal changes
    noise is estimated by the the point to point differences
    signal is estimated by the overall range
    '''
    signal =(np.percentile(y,95)-np.percentile(y,5))
    noise1 = np.percentile(np.abs(np.diff(y)), 99)/2.0
    #noise2 = np.mean(np.abs((y[1:-1]-(y[0:-2]+y[2:])/2)))*4
    #noise3 = noiseEstimate(y)*4
    #print((signal/noise1, signal/noise2, signal/noise3))
    return signal/noise1

def fileAnalyze(f, ramp_nonlinear=False, plot=False, warnings=True, phaseData=0, refSigs = [0,1], ramp_i = 3) :
    '''
    Takes a 4 stream data file (ref, trans, refl, ramp)
    and calculates
    Aratio: ratio of amplitude to ref
    Phase: phase relative to ref
    c = Aratio*exp(i*phase)
    ramp_nonlieanr - attempt to recover the instantaneous phase from the ref signal
    returns: [c_ref, c_trans]
    ramp_i = index of the ramp signal
    refSigs: these are the data indices of the reference signals to be used
            (eg: blue and green pinhole detector sigs)
            The first will be used for the non-linear phase ramp Hilbert transform correction
    '''
    # Load data, set up x index array
    data = np.load(f)
    ix = np.arange(data.shape[1])


    # pre-plots
    if plot :
        snr = [snrEstimate(y) for y in data]
        print('snr = {}'.format(snr))
        plotN = 6
        plt.figure(1, figsize=(8,7), dpi=200)
        plt.clf()
        # Ramp scaled to on ref sig
        plt.subplot(plotN,1,0+1)
        plt.plot((data[ramp_i]-data[ramp_i].min())/(data[ramp_i].max()-data[ramp_i].min())*(data[0].max()-data[0].min())+data[0].min(), label='scaled ramp' )
        # Ref, Refl, trans
        for i in [0,1,2] :
            plt.subplot(plotN,1,i+1)
            plt.plot(data[i], label='signal: {}'.format(i))
            plt.legend()
        plt.show(plotblocking)
        plt.pause(.1)

    # Make sure data is good
    #if snrEstimate(data[0]) < 4 :
    #    if warnings : print(
    #        "Warning, skipping data with bad SNR ({}) : ".format(
    #            snrEstimate(data[0])) + f )
    #    if plot :
    #        plt.show()
    #        plt.pause(.1)
    #        input('Press key to continue')
    #    return 0.0, 0.0

    # Find ramp fit
    ramp_data = data[ramp_i]
    ramp_p, ramp_p0, ramp_success = rampFit(ix, ramp_data, Ncyc=1, symest=.9, plot=False)
    if not ramp_success :
        if warnings : print("Warning, bad ramp : " + f)
        return 0.0, 0.0, 0.0, 0,0

    # Truncate data to useful region (i.e. just the upside or downside of the ramp)
    ramp_T = ramp_p[3]
    ramp_sym = ramp_p[4]
    bufferN = int(ramp_T/10) # Number of points to chop off at beginning and end of region
    upSide = True
    if upSide : # Select the start of the upside of ramp
        istart = int(ramp_p[2]) + bufferN
    else : # Select the start of the downside of ramp
        istart = int(ramp_p[2]) + bufferN + int(ramp_T*ramp_sym)
    ifinish = int(istart + ramp_T*ramp_sym) - 2*bufferN
    istart = max(istart, 0)
    ifinish = min(ifinish, len(ramp_data))
    data = data[:,istart:ifinish]
    ix =     ix[istart:ifinish]
    ref_data = data[refSigs[0]]

    # Detrend oscillating signals
    for ref_i in [0,1,2] :
        data[ref_i] = deTrend(ix,data[ref_i])

    # Clean up ramp: replace with calculated values from model)
    data[ramp_i,:] = rampModel(ix, *ramp_p)

    # Clean up ramp: fix non-linearities using Hilbert transform
    if ramp_nonlinear :
        phase_p = phaseFit(ix, ref_data, plot=False)
        ix_new = (phase_p-phase_p.min())/(phase_p.max()-phase_p.min())*(ix.max()-ix.min())+ix.min()
        ix, data = resampleData(phase_p, data, oversample=1, plot=False)
        ref_data = data[refSigs[0]]


    # Find ref signal fit
    ## Est threshold based on slope at zero crossings + noise estimate
    sig = [0.0,0.0, 0.0, 0.0]
    looppass = 0;
    outeri = 0
    ixT = None
    ref_sin = None
    while outeri < len(refSigs) :
        #1/0  # finish implementing dual wavelength crosstalk removal
        ref_i = refSigs[outeri]
        outeri+=1
        ref_data = data[ref_i]

        max_periods = 10 # estimate of the max periods expected
        minslope = (ref_data.max()-ref_data.min())/ref_data.size*max_periods
        noise = np.percentile(np.diff(ref_data),95)
        thresh = noise/minslope
        thresh = 15
        ref_p, success = sinFitHybrid(ix, ref_data, thresh, plot=False)
        if not success :
            if warnings : print("Warning, sin fit failed.\t" + f )
            sig[ref_i] = 0.0 + 0.0j
            continue

        # Select integer number of wavelengths (as close as possible)
        freq = ref_p[2]
        periods = int((ix.max()-ix.min())*freq)
        window = periods/freq
        ifinish = np.argmin(np.abs((ix-ix[0])-window))
        dataT = data[:,0:ifinish]
        ixT =     ix[0:ifinish]
        if len(ixT) < 2 :
            if warnings : print("Warning, data window cut to zero" + f )
            sig[ref_i] = 0.0 + 0.0j
            continue

        # Perform lock-in
        ## Create ref signals
        ref_p[0] = 0 # Remove ref offset
        ref_A = ref_p[1]
        ref_p[1] = 1 # Make ref signal -1 to 1
        ref_sin = sinModel(ixT, *ref_p)
        # What is this doing??? - alternating +/-pi/2 depending on if upSide is
        # 1/0
        ref_p[3] += (int(upSide)*2-1)*np.pi/2     # a cos ref signal
        ref_cos = sinModel(ixT, *ref_p)

        # Lock in/quadrature measure, divide out amplitude changes (=laser power drift?)
        sig[ref_i] = (np.mean(dataT[2]*ref_sin) + 1j*np.mean(dataT[2]*ref_cos))/ref_A
        sig[ref_i+2] = ref_A

    # Post plots
    if plot :
        print('Ramp start,finish, T: {}'.format( [istart,ifinish,ramp_T]))
        plt.figure(1)
        print("Result: {}, {}".format(sig[0],sig[1]))
        plt.subplot(plotN,1,4)
        plt.plot(ixT,ref_sin, label='sin fit')
        plt.legend()
        plt.plot(ixT,ref_cos, label='cos generate')
        plt.legend()
        plt.plot(ix,(data[ref_i]-data[ref_i].mean())/data[ref_i].std()/np.sqrt(2), '.', ms=1, label='sig {}'.format(ref_i) )
        plt.legend()
        plt.subplot(plotN,1,5)
        plt.plot(phase_p-np.linspace(phase_p[0], phase_p[-1], len(phase_p)), label='nonlinearity in phase ramp')
        plt.ylabel('Radians')
        plt.legend()
        plt.subplot(plotN,1,6)
        obj_i = 2 # indexo of object signal data
        plt.plot(ixT,(dataT[obj_i]-dataT[obj_i].mean())/dataT[obj_i].std()/np.sqrt(2), '.', ms=1, label='phase corrected  sig {}'.format(obj_i) )
        plt.plot(ixT,ref_sin, label='sin fit')
        plt.legend()
        plt.show(plotblocking)
        plt.pause(.1)
        input('Press key to continue')

    return sig[0], sig[1], sig[2], sig[3]

# Wrapper for fileAnalyze for multiprocessing
# Needs tstart, Njobs and Nleft
# job params: (filename, {param dict for fileAnalyze})
# eg. './data015/step_x_-60_y_-60.npy', {'ramp_nonlinear': True, 'plot': False}
def fileAnalyzeWrapper(job_params) :
    global tstart, Njobs, Nleft, poolHalt
    d1,d2,a1,a2 = 0.0, 0.0, 0.0, 0.0
    try :
        d1,d2,a1,a2 = fileAnalyze(job_params[0],**job_params[1]) # this is the thing that takes time

        # Estimate time
        with Nleft.get_lock():
            Nleft.value-=1
            Ndone = Njobs-Nleft.value
            tavg = (time.time()-tstart)/Ndone
            tremaining = int(tavg*Nleft.value)
            print("  Time remaining: {0} min {1} s,  avg: {2:02.3} s/point, on {3}/{4}       ".format(
                int(tremaining/60), tremaining%60, tavg, Ndone, Njobs), end='\r' )
            sys.stdout.flush()
    except KeyboardInterrupt :
        print('Keyboard Interrupt')
        poolHalt.value = True
    except Exception as e :
        #poolHalt.value = True
        print('Error occurred in worker: ')
        print(job_params)
        print(e)
        traceback.print_exc()
        #raise e
    return str(fileIndexPair(job_params[0])),d1,d2, a1,a2

def fileIndexPair(f):
    nx = int(f.replace('.','_').split('_')[-4])
    ny = int(f.replace('.','_').split('_')[-2])
     #should be nx, ny
    return (nx, ny)

def dataAnalyze(dataRunFolder, plot=False, cont=True, save=True, threads=1, resume=True, zoom=1, justload=False, newOnly=True, refSigs=[0,1]) :
    '''
    Opens each npy file in a dataRunFolder (either string or integer):
        - sends file to analysis function
        - adds result to dictionary
        - returns 2 dicts of values
        refl_data = {(nx,ny) = complex_value}
        trans_data = {(nx,ny) = complex_value}
    '''
    # Get list of files
    datafiles = [f for f in os.listdir(dataRunFolder) if f[0:4] == 'step' ]

    # Sort datafiles by y value first, then x value
    #datafiles.sort(key=lambda x:(int(x.replace('.','_').split('_')[4]), int(x.split('_')[2])) )
    datafiles.sort(key=fileIndexPair)

    # Load, or set up data dicts
    data1_dict = {}
    data2_dict = {}
    if cont :
        try :
            with open(dataRunFolder+'data_sig1.json') as f : data1_dict = json.load(f)
            with open(dataRunFolder+'data_sig2.json') as f : data2_dict = json.load(f)
        except (IOError, TypeError, ValueError) as e :
            print('Error reading data')
            print(e)

    # Loop over files and add each that needs to be analyzed
    # to a queue
    Njobs = 0
    Nfiles = len(datafiles)
    file_i = 1
    jobq = []
    jobkeys = []
    print("Checking for files that need to be analyzed. Nfiles = {}".format(len(datafiles)))
    if justload : datafiles = []
    for f in datafiles :
        # Get index
        nx, ny = fileIndexPair(f)

        # Check if data already calculated
        fkey = str((nx,ny))
        print('WARNING: Update (0.0, 0.0) check to be whatever the 0+0j string looks like!')
        if not resume or (
            fkey not in data1_dict or
            fkey not in data2_dict or
            (not newOnly and (data1_dict[fkey]=='(0.0, 0.0)' or
            data2_dict[fkey]=='(0.0, 0.0)')) ) :
            # add file to queue
            jobq.append((dataRunFolder+f, {'ramp_nonlinear':True, 'plot':False,'refSigs':refSigs}, ))
            jobkeys.append(fkey)
            #jobq.put('hi')
            Njobs+=1

    # Process it all
    def pool_init(t, n, nl, pH) :
        global tstart, Njobs, Nleft, poolHalt
        tstart = t
        Njobs = n
        Nleft = nl
        poolHalt = pH

    # Start multiprocessing
    print('Starting processes - njobs: {}'.format(Njobs))
    tstart = time.time()
    Nl = multi.Value('i',len(jobq))
    poolHalt = multi.Value('b', False)
    Nproc = multi.cpu_count()*2
    Nset = 8 # Number of files per process per set
    print('Waiting for threads to finish')
    done = False
    results = []

    Nsave = 500 # Number to calculate before pausing and saving progress
    Nsavei = 0  # counter to keep track of when to save
    while not done :
        try :
            pool = multi.Pool(processes=Nproc, initializer=pool_init, initargs=(tstart, Njobs, Nl, poolHalt) )
            jobq_set = [jobq.pop() for i in range(min(Nproc*Nset,len(jobq)))]
            results_t = pool.map_async(fileAnalyzeWrapper, jobq_set).get(10000000)
            pool.close()
            pool.join()
            results += results_t
            if len(jobq) == 0  or poolHalt.value:
                done = True

            # Save intermediate results
            # Format and save results
            Nsavei+=Nproc*Nset
            if Nsavei > Nsave :
                for result in results :
                    data1_dict[result[0]] = str(result[1])+','+str(result[3])
                    data2_dict[result[0]] = str(result[2])+','+str(result[4])
                with open(dataRunFolder+'data_temp1.json','w') as f :
                    json.dump(data1_dict, f, indent=2)
                with open(dataRunFolder+'data_temp2.json','w') as f :
                    json.dump(data2_dict, f, indent=2)
                Nsavei=0

        except KeyboardInterrupt as e :
            print("Error occured:")
            print(e)
            done = True
    pool.close()
    pool.join()
    print('Done')


    # Format results
    for result in results :
        data1_dict[result[0]] = str(result[1])+','+str(result[3])
        data2_dict[result[0]] = str(result[2])+','+str(result[4])

    # Save results
    if save :
        with open(dataRunFolder+'data_sig1.json','w') as f :
            json.dump(data1_dict, f, indent=2)
        with open(dataRunFolder+'data_sig2.json','w') as f :
            json.dump(data2_dict, f, indent=2)

    return data1_dict, data2_dict

def dataDictToArray(data_dict, params):
    '''
    Takes a data dict and makes a numpy array where
        array[mx,my] = dict['(nx, ny)']
             with m = (n-nfirst)/stepsize
    '''
    kxydata = np.zeros((params['y_N'], params['x_N'] ), dtype=np.complex64)
    kxyamp = np.zeros((params['y_N'], params['x_N'] ), dtype=np.float64)
    Nbad = 0
    for d in data_dict :
        index_x, index_y = [int(i)/params['x_stepN'] for i in d[1:-1].split(',')]
        index_x -= params['x_firstN']
        index_x /= params['x_stepN']
        index_y -= params['y_firstN']
        index_y /= params['y_stepN']
        index_x = int(index_x)
        index_y = int(index_y)
        try:
            s = data_dict[d].split(',')
            kxydata[index_y, index_x] = complex(s[0])
            kxyamp[index_y, index_x] = float(s[1])
        except ValueError as e:
            Nbad+=1
            if data_dict[d]!= '(0.0, 0.0)':
                print('Bad string: ' + data_dict[d])
            kxydata[index_y, index_x] = 0
    print('Bad strings: {}'.format(Nbad))
    print('Update this code when not needed')
    return kxydata, kxyamp

def myImageSave(fname, array, ext='.png', params={'cmap':'bone'}):
    np.save(fname,array)
    if params is 'off' :
        #assert len(array.shape) == 2
        # Change range to 0-256
        array = (array - array[array>-np.inf].min())
        array /= array[array<np.inf].max()
        array[array<0]=0
        array[array>1]=0
        array*=256

    else :
        #Save
        #scipy.misc.imsave(fname+ext, array)
        plt.imsave(fname+ext, array, **params)


def kDataClean(a, average=False, interp=False):
    '''
    Removes 'bad elements' in the data - i.e. points with zero value.
    Helps to make plots look a little nicer - and fix issues with log plotting, etc.

    Default: set all points that are 0 to the average of the points around it.
             Any remaining zeros (i.e. from a large zero region) are instead
             set to be the smallest non-zero value of abs(a)

    average: Set all points to an average of their surrounding pixels.

    interp:  Replaces above functions. Interpolates all zero pixels based on a
            cubic interpolation from all non-zero pixels
    '''
    # Interpolation
    if interp :
        X, Y = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]))
        a[a==0] = scipy.interpolate.griddata((X[a!=0], Y[a!=0]), a[a!=0], (X[a==0],Y[a==0]), method='cubic')
        return a

    # Calculate nearest neighbor average
    b = a*0.0
    for j in (-1,0,1):
        for i in (-1,0,1):
            if not (i==0 and j==0):
                b+=np.roll(a,(i,j))
    b /= 8.0

    # Remove remaining zeros
    nz = b[b==0].size
    if nz > 0 and nz < b.size :
        b[b==0] = np.abs(b[b!=0]).min()

    # Replace 0's (bad values) with nearest neighbor average
    a[a==0.0] = b[a==0]

    # Average everything if desired
    if average :
        a = b

    return a

def momentCalc(x,y,a,n=1) :
    '''
    Calculate the nth moment of 2d data
    x : meshgrid of x values
    y : meshgrid y values
    a : the data to calculate
    n : the order
    return : cx, cy
    '''
    A = np.abs(a)**1
    A /= A.sum()
    cx = (x**n * A).sum()
    cy = (y**n * A).sum()
    return cx, cy

def peakFind(x,y,a, n=5) :
    '''
    Calculate the nth moment of 2d data
    x : meshgrid of x values
    y : meshgrid y values
    a : the data to calculate
    n : the L1 radius to average when picking max value
    return : cx, cy
    '''
    A = np.abs(a)**.5
    # Average
    B = 0
    for j in range(-n,n+1):
        for i in range(-n,n+1):
            B+=np.roll(A,(i,j))
    cy, cx = np.unravel_index(np.argmax(B, axis=None), B.shape)
    return cx, cy


def dataInterpret(dataRunFolder, data_sig1, data_sig2, plotblocking=False, backProjectD=None) :
    '''
    Put data from pair indexed dict into a 2D array
    FFT
    plot
    '''
    params = json.load(open(dataRunFolder+'params_set.json'))
    #params.update(json.load(open(dataRunFolder+'params_derived.json')))
    params.update(deriveParams(params))

    print( 'FOV = ({},{}) mm'.format(params['x_FOV'],     params['y_FOV']) )
    print( 'res = ({},{}) mm'.format(params['x_res'], params['y_res']) )
    print( 'Notes: ' + params['notes'] )
    x_dk = params['x_kres']
    y_dk = params['y_kres']
    x_FOV = params['x_FOV']
    y_FOV = params['y_FOV']
    #aspect = params['y_FOV']/params['x_FOV']*params['x_N']/params['y_N']
    aspect = 'equal'

    # Make an evenly spaced array of rrefl and trans data
    kxydata_sig1, kxyamp_sig1 = dataDictToArray(data_sig1, params)
    kxydata_sig2, kxyamp_sig2 = dataDictToArray(data_sig2, params)
    kxydata_sig1 = kxydata_sig1[:,::-1]
    kxydata_sig2 = kxydata_sig2[:,::-1]

    # Divide out amplitude changes
    if False :
        kxyamp_sig1[kxyamp_sig1==0] = kxyamp_sig1.mean()
        kxyamp_sig2[kxyamp_sig2==0] = kxyamp_sig2.mean()
        kxydata_sig1 /= kxyamp_sig1**1
        kxydata_sig2 /= kxyamp_sig2**1

    # Index grid
    Ny, Nx = kxydata_sig1.shape
    idx, idy = np.meshgrid(np.arange(Ny), np.arange(Nx))

    # Phase correction tests
    if False :
        r = np.sqrt((idx-Nx/2.0)**2+(idy-Ny/2.0)**2)
        pfix = lambda a :  a/(1-.3*r/r.max())
        kxydata_sig1 = np.abs(kxydata_sig1)*np.exp(
            1j*pfix(np.angle(kxydata_sig1)))
        kxydata_sig2 = np.abs(kxydata_sig2)*np.exp(
            1j*pfix(np.angle(kxydata_sig2)))

    # Non-linear correction tests
    if False :
        l = lambda x : x/(1+np.abs(x)**1)
        kxydata_sig1 = l(kxydata_sig1)
        kxydata_sig2 = l(kxydata_sig2)

    # Better center data
    if True:
        # Estimate k-space center using first moment calc
        # Sig 1
        cx, cy = peakFind(idx,idy,kxydata_sig1, n=30)
        rx, ry = (np.r_[cx,cy] - np.r_[idx.shape]/2)
        print('k-space off-center estimate: {}, {}'.format(rx, ry))
        #print(kxydata_sig1, rx)
        #kxydata_sig1 = np.roll(kxydata_sig1, -rx, axis=1) #2+
        #kxydata_sig1 = np.roll(kxydata_sig1, -ry, axis=0) #0+
        kxydata_sig1 = np.roll(kxydata_sig1, -int(rx), axis=1) #2+
        kxydata_sig1 = np.roll(kxydata_sig1, -int(ry), axis=0) #0+
        # Sig 2
        cx, cy = peakFind(idx,idy,kxydata_sig2, n=30)
        rx, ry = (np.r_[cx,cy] - np.r_[idx.shape]/2)
        print('k-space off-center estimate: {}, {}'.format(rx, ry))
        #kxydata_sig2 = np.roll(kxydata_sig2, -rx, axis=1) #2+
        #kxydata_sig2 = np.roll(kxydata_sig2, -ry, axis=0) #1
        kxydata_sig2 = np.roll(kxydata_sig2, -int(rx), axis=1) #2+
        kxydata_sig2 = np.roll(kxydata_sig2, -int(ry), axis=0) #1


    # Filter out some areas
    if False :
        fx, fy = 237, 62 # index coordinates of spot to filter
        r = 15 # radius in pixels
        o = 3.0/4 # 1 = diamond mask, 2 = circle, etc.
        kxydata_sig1[(np.abs(idx-fx)**o+np.abs(idy-fy)**o)<r**o] = 0
        kxydata_sig2[(np.abs(idx-fx)**o+np.abs(idy-fy)**o)<r**o] = 0
        fx, fy = Nx-fx, Ny-fy
        kxydata_sig1[(np.abs(idx-fx)**o+np.abs(idy-fy)**o)<r**o] = 0
        kxydata_sig2[(np.abs(idx-fx)**o+np.abs(idy-fy)**o)<r**o] = 0

    # Filter out some areas
    if False :
        # top bottom crop
        #y1 = 156
        #y2 = 301
        #kxydata_sig1[:,x1:x2] = 0
        #kxydata_sig2[:,x1:x2] = 0
        x1 = 0
        x2 = 146
        kxydata_sig1[:,x1:x2] = 0
        kxydata_sig2[:,x1:x2] = 0

    # Clean up data (remove 0's)
    if True:
        kxydata_sig1 = kDataClean(kxydata_sig1, interp=False, average=False)
        kxydata_sig2 = kDataClean(kxydata_sig2, interp=False, average=False)
    else :
        kxydata_sig1[kxydata_sig1==0] = np.abs(kxydata_sig1[kxydata_sig1!=0]).min()
        kxydata_sig2[kxydata_sig2==0] = np.abs(kxydata_sig2[kxydata_sig2!=0]).min()

    #kxydata_sig1 = kxydata_sig1.transpose()
    #kxydata_sig2 = kxydata_sig2.transpose()

    # Use half of kspace
    kxydata_sig1C = kxydata_sig1.copy() # Conjugate sym
    kxydata_sig2C = kxydata_sig2.copy()
    kxydata_sig1P = kxydata_sig1.copy() # Half of k-space
    kxydata_sig2P = kxydata_sig2.copy()
    NR, NC = kxydata_sig2.shape
    print(NR, NC)
    kxydata_sig1C[int((NR+1)/2):,:] = np.conj(kxydata_sig1C[int(NR/2-1)::-1,::-1]) # Conj sym
    kxydata_sig2C[int((NR+1)/2):,:] = np.conj(kxydata_sig2C[int(NR/2-1)::-1,::-1]) # Conj sym
    #if NR%2 :
    #kxydata_sig1C[152:,:] = np.conj(kxydata_sig1C[149:0:-1,::-1]) # Conj sym
    #kxydata_sig2C[152:,:] = np.conj(kxydata_sig2C[149:0:-1,::-1]) # Conj sym
    #kxydata_sig1C[150,:] = 0
    #kxydata_sig2C[150,:] = 0

    kxydata_sig1P[int(NR/2):,:] = 0 # Just use positive frequencies
    kxydata_sig2P[int(NR/2):,:] = 0 # Just use positive frequencies

    # Reconstruct
    N = kxydata_sig2.shape[0]
    kxydata_sig1[0:int(N/2),:] = (kxydata_sig1[0:int(N/2),:])
    kxydata_sig2[0:int(N/2),:] = (kxydata_sig2[0:int(N/2),:])

    im_sig1  = np.fft.ifftshift(np.fft.ifft2(kxydata_sig1))
    im_sig1C = np.fft.ifftshift(np.fft.ifft2(kxydata_sig1C))
    im_sig1P = np.fft.ifftshift(np.fft.ifft2(kxydata_sig1P))

    im_sig2  = np.fft.ifftshift(np.fft.ifft2(kxydata_sig2))
    im_sig2C = np.fft.ifftshift(np.fft.ifft2(kxydata_sig2C))
    im_sig2P = np.fft.ifftshift(np.fft.ifft2(kxydata_sig2P))
    imgs = [im_sig2, im_sig1, im_sig2C, im_sig2P, im_sig1C, im_sig1P]

    # Plot
    plotNr=3
    plotNc=2
    plt.close(1)
    plt.figure(1, dpi=100, figsize = (8,14))
    pi = 1

    # Plot 2d k data
    cmap = 'inferno'
    extent = (0, x_FOV, 0, y_FOV)
    kplotparams = {'norm':pltcolors.LogNorm(), 'aspect':aspect, 'cmap':cmap,
                   'interpolation':'nearest'}
    imparams = {'cmap':cmap}

    plt.subplot(plotNr,plotNc,pi); pi+=1
    a = np.abs(kxydata_sig2)
    #a = np.abs(kxydata_sig1C)
    name = 'arrIm_kspace_log_1'
    plt.imshow(a, **kplotparams)
    plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
    #myImageSave(dataRunFolder+dataRunFolder[2:-1] + '_' + name, np.log10(a+10e-16), params=imparams)
    myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, np.log10(a+10e-16), params=imparams)

    plt.subplot(plotNr,plotNc,pi); pi+=1
    a = np.abs(kxydata_sig1)
    #a = np.abs(kxydata_sig1P)
    name = 'arrIm_kspace_log_2'
    plt.imshow(a, **kplotparams)
    plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
    #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, np.log(a+10e-16), params=imparams)
    myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, np.log(a+10e-16), params=imparams)

    # Shift image
    if 'img_x_shift' in params :
        imgs = [np.roll(im, params['img_x_shift'], axis=1) for im in imgs]
        im_sig2, im_sig1, im_sig2C, im_sig2P, im_sig1C, im_sig1P = imgs
    if 'img_y_shift' in params :
        imgs = [np.roll(im, params['img_y_shift'], axis=0) for im in imgs]
        im_sig2, im_sig1, im_sig2C, im_sig2P, im_sig1C, im_sig1P = imgs
    if 'img_x_mirror' in params and params['img_x_mirror'] :
        imgs = [im[:,::-1] for im in imgs]
        im_sig2, im_sig1, im_sig2C, im_sig2P, im_sig1C, im_sig1P = imgs
    if 'img_y_mirror' in params and params['img_y_mirror'] :
        imgs = [im[::-1,:] for im in imgs]
        im_sig2, im_sig1, im_sig2C, im_sig2P, im_sig1C, im_sig1P = imgs

    # Plot object space
    cx, cy = x_FOV/2, y_FOV/2
    if zoom :
        z = 2
        assert z > 0, 'invalid zoom'
    else :
        z = 1

    plt.subplot(plotNr,plotNc,pi); pi+=1
    a = np.abs(im_sig1)
    name = 'arrIm_image_1'
    plt.imshow(a, aspect=aspect, extent=extent, interpolation='nearest')
    plt.text(0, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
    #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, a)
    myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, a)
    if zoom : plt.xlim(xlim); plt.ylim(ylim);

    xlim, ylim = [cx-x_FOV/2/z,cx+x_FOV/2/z], [cy-y_FOV/2/z,cy+y_FOV/2/z]
    #extent = (0,xlim, 0, ylim)
    plt.subplot(plotNr,plotNc,pi); pi+=1
    a = np.abs(im_sig2)
    name = 'arrIm_image_2'
    plt.imshow(a, aspect=aspect, extent=extent, interpolation='nearest')
    plt.text(0, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
    #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, a)
    myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, a)
    if zoom : plt.xlim(xlim); plt.ylim(ylim);
    print(xlim)

    #plt.subplot(plotNr,plotNc,pi); pi+=1
    #plt.imshow(np.abs(im_sig23), aspect=aspect, extent=extent, interpolation='nearest')
    #plt.text(0, 1, 'image 1, half k', transform=plt.gca().transAxes, backgroundcolor='white' )

    extraPlots = ['phase',  'conj', 'pos', '1_','2_', 'backProject']
    if backProjectD is not None :
        extraPlots = extraPlots[5]
    else : extraPlots = extraPlots[0]
    # Phase images
    myphasemap = make_anglemap( N = 256, use_hpl = True )
    phase_cutoff_perc = 99 #  99  for res targets
    phase_cutoff_rat = .35 # 1/4 for res targets, 0 to turn off
    im_sigs = [im_sig1, im_sig2]
    for i in [0,1] :
        im = im_sigs[i]
        a = np.angle(im)
        a = myunwrap(a)+np.pi/2
        b = np.abs(im)
        phasemap = myphasemap
        a[b<np.percentile(b, phase_cutoff_perc)*phase_cutoff_rat] = 0*np.pi/2 # Black out low value areas
        #a[b<b.max()/4] = 0*np.pi/2 # Black out low value areas
        name = 'arrIm_image_{}_phase'.format(i+1)
        #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, a, params={'cmap':phasemap})
        myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, a, params={'cmap':phasemap})
        if extraPlots is 'phase' : # Plot phase
            plt.subplot(plotNr,plotNc,pi); pi+=1
            plt.imshow(a, aspect=aspect, extent=extent, interpolation='nearest', cmap = phasemap)
            plt.text(0, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )

    # Conjugate sym image
    for im, nm in [[im_sig1C, "1_conj_sym"],
                   [im_sig1P, "1_posKonly"],
                   [im_sig2C, "2_conj_sym"],
                   [im_sig2P, "2_posKonly"]] :
        a = np.abs(im)
        name = 'arrIm_image_'+nm
        #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, a)
        myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, a)
        if extraPlots in nm :
            # Mean weighted angle
            mwa = (np.abs(np.angle(im*np.exp(1j*np.pi/2)))*np.abs(im)).mean()/np.abs(im).mean()/np.pi-.5
            print(nm+' angle(mean): {:3.3f} mean: {:3.3f} mean: {:3.3f} w|angle|: {:3.3f}'.format(
                np.angle(im.mean()), im.mean()/np.abs(im.mean()), np.imag(im).mean()/np.abs(im.mean()), mwa))

            plt.subplot(plotNr,plotNc,pi); pi+=1
            plt.imshow(a, aspect=aspect, extent=extent, interpolation='nearest', cmap = 'bone')
            plt.text(0, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
    # Back Projection
    if extraPlots is 'backProject' :
        for i, lbda in [ (0, 407.0*10**-6),(1, 532.0*10**-6)] :
            im = im_sigs[i]
            px = params['x_res'] # px size in mm
            if i == 0 :
                px *= 407.0/532.0 # corrected for blue
            lbda_px = 1.0*lbda/px # wavelength in mm -> px
            k_lbda_px = 2*np.pi/lbda_px # wavenumber in 1/px
            if backProjectD is None :
                d = 156 # 155 tried 145-155, similar results
            else : d = backProjectD
            d_px = d/px # distance to object in mm->px

            # Trim im for speed
            #imt = im[50:-50:1, 50:-50:1]
            #imt = im
            #im = imt.copy()

            # 78-80: angle (.020, .020), d~150
            # 72: angle (.020, .021), d=~150, 254, 284?e
            angle = (.020, .021)

            im2 = backProjection(im, d_px, k_lbda_px, r=None, skip=1,
                                 c=(140,190), angle=angle ) #r=50, for 67
            # Object plot
            a = np.abs(im2)

            # Phase plot
            b = np.angle(im2)
            b = myunwrap(b)+np.pi/2
            phasemap = myphasemap
            b[a<np.percentile(a, phase_cutoff_perc)*phase_cutoff_rat] = 0*np.pi/2 # Black out low value areas

            # Save back projected object
            name = 'arrIm_image_{}_backProject_{}'.format(i+1,d)
            #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, a)
            myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, a)

            # Save back projected phase
            name = 'arrIm_image_{}_backProject_phase_{}'.format(i+1,d)
            #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_'+name, b, params={'cmap':phasemap})
            myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_' + name, b, params={'cmap':phasemap})

            plt.subplot(plotNr,plotNc,pi); pi+=1
            plt.imshow(a, aspect=aspect, extent=extent, interpolation='nearest', cmap = 'bone')
            plt.text(0, 1, 'BackProjection', transform=plt.gca().transAxes, backgroundcolor='white' )

    # Finish plot
    plt.tight_layout()
    #plt.draw()
    plt.savefig(dataRunFolder+dataRunFolder[2:-1]+'_'+'reconstruction.pdf')
    #mgr = plt.get_current_fig_manager()
    #mgr.window.SetPosition((1700,100))

    # Dual wavelength
    green = np.abs(im_sig2)
    blue = np.abs(im_sig1)
    green/=green.max()
    blue /= blue.max()
    lblue = 407.0;
    lgreen = 532.0;
    greenZ = scipy.ndimage.zoom(green, 1.0*lgreen/lblue)
    greenZ -= greenZ.min()
    padY = greenZ.shape[0] - green.shape[0]
    padX = greenZ.shape[1] - green.shape[1]
    greenZ = greenZ[int(np.floor(padY/2.0)):-int(np.ceil(padY/2.0)), int(np.floor(padX/2.0)):-int(np.ceil(padX/2.0))]

    # Determine shift, and move green to correspond
    G = np.fft.fft2(greenZ)
    B = np.conjugate(np.fft.fft2(blue))
    ccor = np.fft.fftshift(np.real(np.fft.ifft2(G*B))) # Cross correlation
    shifty, shiftx = np.unravel_index(np.argmax(ccor), G.shape)
    shiftx -= int(G.shape[1]/2)+1
    shifty -= int(G.shape[0]/2)+1
    print('Zoomed green image offset from blue by: {},{} pixels'.format(shiftx, shifty))
    greenZ = np.roll(greenZ, (-shiftx, -shifty), (1,0))
    greenZ = greenZ * blue.mean()/greenZ.mean()

    # Make RGB image
    rgb = np.r_[[blue*0,greenZ,blue]].transpose((1,2,0))
    rgb /= 1.0*rgb.max()
    #myImageSave(dataRunFolder+dataRunFolder[2:-1]+'_rgb', rgb)
    myImageSave(dataRunFolder + '/' + dataRunFolder[-5:-2] + '_rgb', rgb)
    if False :
        plt.figure(2); pi=1;
        plt.subplot(2,2,pi); pi+=1
        name='blue'
        plt.imshow(blue, label=name)
        plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
        plt.subplot(2,2,pi); pi+=1
        name = 'green'
        plt.imshow(green, label=name)
        plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
        plt.subplot(2,2,pi); pi+=1
        name = 'green zoomed'
        plt.imshow(greenZ, label=name)
        plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
        plt.subplot(2,2,pi); pi+=1
        name = 'rgb'
        plt.imshow(rgb, label=name)
        plt.text(.3, 1, name, transform=plt.gca().transAxes, backgroundcolor='white' )
        plt.tight_layout()
        plt.savefig(dataRunFolder+dataRunFolder[2:-1]+'_'+'recon_color.pdf')

    plt.show(plotblocking)
    plt.pause(.1)

if __name__ == '__main__' :
    # Command line parameters
    plotblocking = False
    plotonly = False
    resume = True
    restart = False
    zoom = False
    justplot=False
    resume = True
    analyzeOnly = False
    newOnly = False
    refSigs=[0, 1]
    backProjectD=None
    if '-h' in sys.argv or len(sys.argv) < 2:
        print('python analyze.py dataRun [option1 ... optionN]')
        print('dataRun       : an integer')
        print('-s [filename] : analyze single file')
        print('-z [#]      : only use reference # x')
        print('-b            : enable blocking plot, i.e. plt.show(True)')
        print('-i            : enable interactive plotting')
        print('-p            : just plot')
        print('-r            : restart (and save backup)')
        print('-z            : zoom in by factor of 2')
        print('-a            : analyze only')
        print('-n            : new points only - don\'t look previously failed points')
        sys.exit()
    if '-i' in sys.argv :
        plt.ion()
    if '-a' in sys.argv :
        analyzeOnly=True
    if '-b' in sys.argv :
        plotblocking=True
    if '-p' in sys.argv :
        justplot = True
    if '-r' in sys.argv :
        restart=True
    if '-z' in sys.argv :
        zoom = 2
    if '-n' in sys.argv :
        newOnly = True
    if '-z' in sys.argv :
        refSigs = [int(sys.argv[sys.argv.index('-z')+1])]
    if '-B' in sys.argv :
        backProjectD = int(sys.argv[sys.argv.index('-B')+1])
    # Single File
    if '-s' in sys.argv :
        f = sys.argv[sys.argv.index('-s')+1]
        print("Single analyze: " + f)
        fileAnalyze(f, plot=True, ramp_nonlinear=True, refSigs = refSigs)
        plt.pause(1)
        sys.exit()

    # Full data run
    print('swap ny, nx in fileIndexPair')
    try :
        dataRunFolder = './data{:03}/'.format(int(sys.argv[1]))
    except ValueError :
        dataRunFolder = sys.argv[1]

    print('Using data folder : ' + dataRunFolder)
    #myinput('Press any key to continue, Ctrl-c to cancel')

    # Move previous data and restart
    sig1File = dataRunFolder+'data_sig1.json'
    sig2File = dataRunFolder+'data_sig2.json'
    pdfFile = dataRunFolder+dataRunFolder[:-1]+'_reconstruction.pdf'
    if restart  and os.path.isfile(sig1File) :
        i=0
        while True:
            i += 1
            newfile1 = sig1File[:-5]+'_{}.json'.format(i)
            newfile2 = sig2File[:-5]+'_{}.json'.format(i)
            newfile3 = pdfFile[:-5]+'_{}.pdf'.format(i)
            if not os.path.isfile(newfile1):
                break
        os.rename(sig1File,newfile1)
        os.rename(sig2File,newfile2)
        os.rename(pdfFile, newfile3)

    data_sig1, data_sig2 = dataAnalyze(dataRunFolder, plot=False, save=True,
                                       resume=resume, zoom=zoom, justload=justplot,
                                       newOnly=newOnly, refSigs = refSigs)
    if not analyzeOnly :
        dataInterpret(dataRunFolder, data_sig1, data_sig2, plotblocking=plotblocking, backProjectD=backProjectD)
