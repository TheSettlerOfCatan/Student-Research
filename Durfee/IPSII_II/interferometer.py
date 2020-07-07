import time
import curses
from threading import Thread
import numpy as np

# Assuming you start with mirrors and beams centered:
# Mirror range is a about +/- 32 rotations (32 * 10**12 steps
# Interferometer range is a about +/- 25 rotations before the beam moves off the mirror

beagle = True

if beagle :
    import Adafruit_BBIO.GPIO as GPIO # for beaglebone
else :
    import RPi.GPIO as GPIO # For rpi
    GPIO.setmode(GPIO.BCM) # For rpi

class StepperMotor(object) :
    '''
    Controls a single stepper motor
    '''

    # Static dict of running motors and their corresponding threads
    runningMotors = {}

    def __init__(self, GPIOpins, driveStyle=1, backlash_correct = False, backlash_steps = 300, backlash_dir = 1) :
        '''
        Initializes current position to '0'

        driveStyle:
            (which stepper driver board you are using)
            1 = DRV
            2 = ULN2003AN

        GPIOpins:
            array of rpi pins numbers connected to  stepper
            There should be 2 for stepper style 1, 4 for style 2)

        backlash_correct:
            makes sure to always end movement from the same direction
            to minimize effects from backlash (i.e. when moving in one
            direction, overshoot and come back to the correct location

        backlash_steps:
            Should larger then the maximum amount of backlash is in motor steps
            This should be determined experimentally
        backlash_dir:
            The direction to always come from when correcting for backlash

        '''
        # Initialize attributes
        self.position = 0
        self.backlash_correct = backlash_correct
        self.backlash_steps = backlash_steps
        self.backlash_dir = backlash_dir
        self.GPIOpins = GPIOpins

        # Pick correct driving method based on the board that's hooked up
        if driveStyle == 1 :
            assert len(GPIOpins) == 2, 'Wrong number of pins'
            self.__driveStep = self.__driveStep1
            self.__pulse_length = 1/1000.0
            self.__step_period = 1/1000.0

        elif driveStyle == 2 :
            assert len(GPIOpins) == 4, 'Wrong number of pins'
            self.__driveStep = self.__driveStep2
            self.turnOff = self.__turnOff2
            self.turnOn = self.__turnOn2
            self.__step_period = .75 * 1/1000.0 # must be >= .5 ms
            self.__coil_phase = 0

        else :
            raise ValueError('Invalid driverstyle selection')

        # Initialize Output pins
        for pin in GPIOpins : GPIO.setup(pin, GPIO.OUT)


        # Preload the gears so we know where home is without backlash
        if backlash_correct :
            self.stepHome()

        self.threadedMove = True

    # Private step functions for different stepper connection types

    # DRV8825
    # GPIOpins = [direction_pin, step_pin]
    def __driveStep1(self, nsteps) :

        # Set direction
        if nsteps > 0 : d = 1
        else : d = 0
        GPIO.output(self.GPIOpins[0], d)

        # Move nsteps steps
        for i in range(abs(nsteps)) :
            # Pulse step
            GPIO.output(self.GPIOpins[1], GPIO.HIGH)
            time.sleep(self.__pulse_length)
            GPIO.output(self.GPIOpins[1], GPIO.LOW)

            # Wait an appropriate amount of time
            time.sleep(self.__step_period)

    # UNL2003
    def __driveStep2(self, nsteps) :
        # Steps
        phase_steps = [ [0,0,0,1],
                        [0,0,1,1],
                        [0,0,1,0],
                        [0,1,1,0],
                        [0,1,0,0],
                        [1,1,0,0],
                        [1,0,0,0],
                        [1,0,0,1]]



        # Set direction
        if nsteps > 0 : d = 1
        else : d = -1

        # Move nsteps steps
        for i in range(abs(nsteps)) :
            self.__coil_phase += d
            self.__coil_phase %= 8

            # Energize appropriate coils
            for coil in [0,1,2,3] :
                GPIO.output(self.GPIOpins[coil], phase_steps[self.__coil_phase][coil])
            time.sleep(self.__step_period)

    def __turnOff2(self) :
        for coil in [0,1,2,3] :
            GPIO.output(self.GPIOpins[coil], 0)

    def __turnOn2(self) :
        for coil in [0,1,2,3] :
            GPIO.output(self.GPIOpins[coil], 1)



    def __stepInThread(self, nsteps) :
        '''
        step nsteps
        direction given by sign of nsteps
        '''
        assert isinstance(nsteps, int), 'Stepper movements should be given in integers'
        if self.backlash_correct and self.backlash_dir*nsteps < 0 :
            # Overstep
            self.__driveStep(nsteps-self.backlash_dir*self.backlash_steps)

            # Correct
            self.__driveStep(self.backlash_dir*self.backlash_steps)

            # Remove myself from list of running motors
            if self in StepperMotor.runningMotors :
                del StepperMotor.runningMotors[self]
            else :
                print("Warning: I'm (unexpectedly) already not in the list of running motors")

        else :
            self.__driveStep(nsteps)


    def step(self, nsteps) :

        # Make sure the current motor is not already in use
        # if it is, wait for that thread to end
        if self in StepperMotor.runningMotors :
            StepperMotor.runningMotors[self].join()

        # Warn if stepping a high amount
        if abs(nsteps) > 4 * 2**12 :
            print('\nWarning: going for {} steps ({} rotations)'.format(nsteps,nsteps/2.0**12))
        assert nsteps < 2**19, "Can't step that far!"

        # If threading is enabled, Create a new thread to run the motor
        # otherwise, jsut run the stepping function
        if self.threadedMove :
            thread = Thread(target = self.__stepInThread, args = (nsteps,))
            StepperMotor.runningMotors[self] = thread
            thread.start()
        else :
            self.__stepInThread(nsteps)
        self.position += nsteps


    def stepHome(self ) :
        '''
        Goes to the '0' location
        '''
        self.step(-self.position)

    def stepTo(self, target_position) :
        '''
        Goes to given absolute location (relative to home)
        '''
        self.step(target_position - self.position)


    def resetHome(self) :
        '''
        Sets the current location as 0, or 'home'
        '''
        self.position = 0


class Mirror(object) :
    '''
    Controls a single mirror
        This class keeps track of mirror position in radians.
        The position (in integer steps) of the motors is kept updated to be as
        close as possible to the current set angular position.  This is so cutoff
        errors don't accrue when using angles to move around instead of integer steps

    stepAngleH = angle in degrees corresponding to 1 step of the Horizontal axis
    stepAngleV = angle in degrees corresponding to 1 step of the vertical axis
    rotationN = number of steps for 1 complete rotation
    rotationAngle = angle adjustment from a single screw rotation, in degrees
    '''
    def __init__(self, stepperH, stepperV) :
        '''
        stepperH and stepperV should be StepperMotor objects corresponding
        to the horizontal and vertical axes of the mirror
        '''

        # Initialize motors
        self.stepperH = stepperH
        self.stepperV = stepperV

        # Calculate angle
        # pivot to pivod = 2.25 in
        # thread = 254 TPI
        # rotation angle = arctan( 1/254/ 2.25) ~ 1/(254*2.25)
        self.rotationN = 2**12 # 4096
        self.rotationAngle = 1.0/(254*2.25)
        self.stepAngleH = self.rotationAngle * 1.0/self.rotationN
        self.stepAngleV = self.rotationAngle * 1.0/self.rotationN
        self.angleH = 0
        self.angleV = 0

    def __move(self) :
        '''
        Updates the physical angle to match the programmed angle
        Moves the steppr motors to the position corresponding to the currently set angle
        Should always be called immediately after changing the set angle
        '''
        nH = int(self.angleH / self.stepAngleH)
        nV = int(self.angleV / self.stepAngleV)
        self.stepperH.stepTo(nH)
        self.stepperV.stepTo(nV)

    def angleAdd(self, angleH, angleV) :
        '''
        Moves the mirror by the specified angle in each axis
        '''
        self.angleH += angleH
        self.angleV += angleV
        self.__move()

    def angleSet(self, angleH=None, angleV=None) :
        '''
        Moves the mirror to the specified angle in each axis
        '''
        if angleH != None :
            self.angleH = angleH
        if angleV != None :
            self.angleV = angleV
        self.__move()


    def angleUpdate(self) :
        '''
        Updates the angle counter to match the motor stepper counters
        (Useful if you directly step the stepper object)
        '''
        self.angleH = self.stepperH.position*self.stepAngleH
        self.angleV = self.stepperV.position*self.stepAngleV

    def resetHome(self) :
        '''
        Sets the current position as the 'home' position
        '''
        self.stepperH.resetHome()
        self.stepperV.resetHome()
        self.angleUpdate()

    def turnOff(self) :
        self.stepperV.turnOff()
        self.stepperH.turnOff()


class MirrorPair(object) :
    '''
    Represents one arm of the interferometer
    d_mirror = distance between mirror1 and mirror2
    d_object = distance between mirror2 and object (i.e. 'focal plane')
    d_ratio = d_object/d_mirror (DEPRECATED - use calibration instead)
    calibration = [h1, v1, c1, h2, v2, c2] ~= [-2,0,0,0,-2,0]
    '''
    # Calculations: mirror1 to mirror2 (d1), and mirror2 to spot (d2)
    # given a final angle you want the laser to hit the spot, what to
    # set the angle of the individual mirrors?
    # th1 = -d2/d1 th_f = (1-m) th_f
    # th2 = (1+d2/d1) th_f = m th_f
    # m = (1+d2/d1)
    #
    # Proper calibration:
    # calibration = [h1, v1, c1, h2, v2, c2]
    # see calibration.py and calibration.nb for calculations
    def __init__(self, mirror1, mirror2, calibration=[-2,0,0,-2], d_ratio=None) :
        if d_ratio :
            self.calibration = [-(1+d_ratio),0,0,0,-(1+d_ratio),0]
            print("WARNING: d_ratio deprecated")
        else :
            self.calibration = calibration
        self.mirror1 = mirror1
        self.mirror2 = mirror2
        self.beamAngleH = 0.0
        self.beamAngleV = 0.0

    def angleAdd(self, beamAngleH, beamAngleV) :
        '''
        moves
        '''
        h1, v1, h2, v2 = self.calibration
        c1, c2 = 0.0, 0.0
        th, tv = beamAngleH, beamAngleV
        a = -((c1-th-c2*v1+tv*v1+c1*v2-th*v2)/(1+h1-h2*v1+v2+h1*v2))
        b = -((-c2-c2*h1+c1*h2-h2*th+tv+h1*tv)/(-1-h1+h2*v1-v2-h1*v2))
        a /= 2.0 # beam angle is double mirror angle
        b /= 2.0
        self.mirror1.angleAdd(a, b)
        self.mirror2.angleAdd(h1*a+v1*b+c1,h2*a+v2*b+c2)
        self.beamAngleH += beamAngleH
        self.beamAngleV += beamAngleV

    def angleSet(self, beamAngleH=None, beamAngleV=None) :
        '''
        Moves either or both mirrors to set the beam angle
        to the specified angle in each axis
        '''
        addH = addV = 0
        if beamAngleH != None :
            addH = beamAngleH-self.beamAngleH
        if beamAngleV  != None :
            addV = beamAngleV-self.beamAngleV

        self.angleAdd(addH, addV)

    def positionAdd(self, x, y, d_mirrors=220) :
        '''
        shifts the beam position by about (x,y) mm
        without changing angle

        d_mirrors is distance between the two mirrors i mm
        setting it is necessary fo the shift to be accurate (in mm)

        only valid for small shifts
        '''
        d_mirrors=float(d_mirrors)
        angleH = 1.0*x/d_mirrors
        angleV = 1.0*y/d_mirrors

        self.mirror1.angleAdd(angleH, angleV)
        self.mirror2.angleAdd(-angleH, -angleV)



    def resetHome(self) :
        '''
        Sets the current position as the 'home' position
        '''
        self.mirror1.resetHome()
        self.mirror2.resetHome()
        self.beamAngleH=0
        self.beamAngleV=0

    def turnOff(self) :
        self.mirror1.turnOff()
        self.mirror2.turnOff()

    @property
    def angle(self) :
        return np.array([self.beamAngleH, self.beamAngleV])


#arm2 22.3, 23.1
#arm1 22.2, 11.8+13.1=24.9
class Interferometer(object) :
    '''
        An interferometer
    '''
    def __init__(self) :

        mH = StepperMotor(['P8_28','P8_30','P8_32','P8_34'], driveStyle=2)
        mV = StepperMotor(['P8_27','P8_29','P8_31','P8_33'], driveStyle=2)
        mirror1 = Mirror(mH, mV)

        mH = StepperMotor(['P9_12','P9_14','P9_16','P9_18'], driveStyle=2)
        mV = StepperMotor(['P9_11','P9_13','P9_15','P9_17'], driveStyle=2)
        mirror2 = Mirror(mH, mV)

        mH = StepperMotor(['P8_8','P8_10','P8_12','P8_14'], driveStyle=2)
        mV = StepperMotor(['P8_7','P8_9','P8_11','P8_13'], driveStyle=2)
        mirror3 = Mirror(mH, mV)

        mH = StepperMotor(['P9_21','P9_23','P9_25','P9_27'], driveStyle=2)
        mV = StepperMotor(['P9_22','P9_24','P9_26','P9_28'], driveStyle=2)
        mirror4 = Mirror(mH, mV)

        self.mirrors = [mirror1, mirror2, mirror3, mirror4]
        self.motors = [];
        for m in self.mirrors :
            self.motors += [m.stepperH, m.stepperV]

        # DEPRECATED
        ## Measurements
        #arm1d1 = 22.2      # distance mirror1 to mirror2
        #arm1d2 = 11.8 + 13.1 # distance mirror2 to splitter + splitter to object
        #arm2d1 = 22.3      # distance mirror1 to mirror2
        #arm2d2 = 23.1     # distance mirror2 to splitter + splitter to object
        ## Setup mirrors to coordinate
        #self.arm1 = MirrorPair(mirror1, mirror2, d_ratio=1.0*arm1d2/arm1d1)
        #self.arm2 = MirrorPair(mirror3, mirror4, d_ratio=1.0*arm2d2/arm2d1)

        # Setup mirrors with full calibration numbers copied from
        # calibration.py output
        cal1 = [-1.96919,-0.08679,-0.02537,-1.94263]
        cal2 = [-1.97270,-0.01234,+0.17970,-1.96020]
        
        #cal1 = [-2.15670,-0.11666,+0.01194,-2.04409]
        #cal2 = [-2.18170,-0.07665,-0.01334,-2.02837]

        #cal1 = [-2.191,0.0,+0.0,-2.061] #Remove interdependence
        #cal2 = [-2.191,0.0,-0.0,-2.051] #Remove interdependence
        self.arm1 = MirrorPair(mirror1, mirror2, calibration=cal1)
        self.arm2 = MirrorPair(mirror3, mirror4, calibration=cal2)
        self.singleArm = 0 # 0 to turn singleArm mode off, 1 for arm 1, 1 for arm 2

        # Calculations
        # Fringe spacing: delx = 2l/sin(th)
        wavelength = 532e-6 # mm
        self.angleK = 1.0/(wavelength) # K/angle ratio in lines/mm, small angles
        self.backlashAngle = .001 # .0004 should be enough...

        # Keep track of angle
        self.angleH =0
        self.angleV =0

    def angleAdd(self, angleH, angleV) :
        if self.singleArm == 0 :
            self.arm1.angleAdd(angleH/2.0,angleV/2.0)
            self.arm2.angleAdd(-angleH/2.0,-angleV/2.0)
        if self.singleArm == 1 :
            self.arm1.angleAdd(angleH,angleV)
        if self.singleArm == 2 :
            self.arm2.angleAdd(-angleH,-angleV)

        # Update Angle
        self.angleH += angleH
        self.angleV += angleV

    def angleSet(self, angleH=None, angleV=None) :
        arm1H = arm2H = arm1V = arm2V = None
            
        # Each arm makes 1.2 the angle, if both are moving
        m = 1.0/2.0 
        if self.singleArm != 0 :
            m = 1.0

        if angleH != None :
            arm1H = angleH*m
            arm2H = -angleH*m
            self.angleH = angleH

        if angleV != None :
            arm1V = angleV*m
            arm2V = -angleV*m
            self.angleV = angleV

        if self.singleArm != 2 :
            self.arm1.angleSet(arm1H,arm1V)
        if self.singleArm != 1 :
            self.arm2.angleSet(arm2H,arm2V)




    def turnOff(self) :
        self.arm1.turnOff()
        self.arm2.turnOff()

    def resetHome(self) :
        self.arm1.resetHome()
        self.arm2.resetHome()

        # Keep track of angle
        self.angleH =0
        self.angleV =0

    def kAdd(self, kx, ky):
        '''
        Adds to the wavenumber in lines/mm (i.e. NOT rad/mm)
        '''
        self.angleAdd(kx/self.angleK, ky/self.angleK)

    def kSet(self, kx=None, ky=None):
        '''
        Goes to to the wavenumber in lines/mm (i.e. NOT rad/mm)
        '''
        xangle, yangle = None, None
        if kx != None :
            xangle = kx/self.angleK
        if ky != None :
            yangle = ky/self.angleK
        self.angleSet(xangle, yangle)

    def positionAdd(self, x, y, d_mirrors=220) :
        '''
        Moves the output position/angle, without changing the angle
        between beams, by about (x,y) mm

        d_mirrors is distance between the two mirrors i mm
        setting it is necessary fo the shift to be accurate (in mm)
        '''
        self.arm1.positionAdd(x,y,d_mirrors)
        self.arm2.positionAdd(x,y,d_mirrors)

    def backlashPositiveH(self) :
        '''
        Loads in the positive Horizontal direction (adds negative angle, 
        then positive to get back to where it was at)
        '''
        self.angleAdd(-self.backlashAngle, 0)
        self.angleAdd(self.backlashAngle, 0)
    def backlashNegativeH(self) :
        '''
        Loads in the negative Horizontal direction (adds positive angle, 
        then negative to get back to where it was at)
        '''
        self.angleAdd(self.backlashAngle, 0)
        self.angleAdd(-self.backlashAngle, 0)
    def backlashPositiveV(self) :
        '''
        Loads in the positive vertical direction (adds negative angle, 
        then positive to get back to where it was at)
        '''
        self.angleAdd(0,-self.backlashAngle)
        self.angleAdd(0,self.backlashAngle)
    def backlashNegativeV(self) :
        '''
        Loads in the negative vertical direction (adds positive angle, 
        then negative to get back to where it was at)
        '''
        self.angleAdd(0,self.backlashAngle)
        self.angleAdd(0,-self.backlashAngle)
    def backlashPositive(self) :
        '''
        Loads in the positive diagonal direction (adds negative angle, 
        then positive to get back to where it was at)
        '''
        self.angleAdd(-self.backlashAngle, -self.backlashAngle)
        self.angleAdd(self.backlashAngle, self.backlashAngle)
    def backlashNegative(self) :
        '''
        Loads in the negative diagonal direction (adds positive angle, 
        then negative to get back to where it was at)
        '''
        self.angleAdd(self.backlashAngle, self.backlashAngle)
        self.angleAdd(-self.backlashAngle, -self.backlashAngle)

    def threadingOn(self):
        '''
        Turn on threading, so all motors move simultaneously
        '''
        for m in self.motors:
            m.threadedMove = True

    def threadingOff(self):
        '''
        Turn off threading, so only one motor will move at a time
        '''
        for m in self.motors:
            m.threadedMove = False

    def setSingleArm(self, arm=2) :
        '''
        Put the interferometer in single arm mode (all beam angle
        changes are done by moving the mirrors in just one arm)

        arm : the one you want to move
        '''
        self.singleArm = arm

    @property
    def angle(self) :
        return self.arm2.angle-self.arm1.angle

def manualMove(obj, incr=.001, move=None, message='') :
    '''
    Listens to keyboard input and moves the object (eg. mirror) accordingly
    the object must have 'addAngle(x,y)' and 'turnOff()' functions
    you may also pass in a seperate 'move' function (accepting
    an x and y parameter) to use in place of the angleAdd function

    w,a,s,d for coarse movements,
    i,j,k,l for fine (coarse/10)
    q to quit

    incr sets the coarse movement corresponding to a single key press

    Pass in an object with angleAdd and turnOff functions
    '''
    incr = float(incr)
    if move == None :
        if not hasattr(obj, 'angleAdd') :
            print('Must have angleAdd function')
            return
        move = obj.angleAdd
    if not hasattr(obj, 'turnOff') :
        print('Must have turnOff function')
        return


    screen = curses.initscr() # get the curses screen window
    #curses.noecho() # turn off input echoing
    curses.cbreak() # respond to keys immediately (don't wait for enter)

    print(message+'\n\r')
    print('w,a,s,d for coarse movements\n\r')
    print('i,j,k,l for fine movements (coarse/10)\n\r')
    print('+ or - to change base sensitivity \n\r')
    print('q to quit\n\r')

    try:
        while True:
            char = screen.getch()
            if char == ord('q'):
                break
            elif char == ord('a') :
                move(-incr,0)
            elif char == ord('d') :
                move(incr,0)
            elif char == ord('w') :
                move(0,+incr)
            elif char == ord('s') :
                move(0,-incr)
            elif char == ord('j') :
                move(-incr/10.0,0)
            elif char == ord('l') :
                move(incr/10.0,0)
            elif char == ord('i') :
                move(0,+incr/10.0)
            elif char == ord('k') :
                move(0,-incr/10.0)
            elif char == ord('b') :
                print('backlash loading (negative to positive)\r\n')
                move(-incr, -incr)
                move(incr, incr)
            elif char == ord('+') :
               incr*=10
               print('sensitivity = {}\n\r'.format(incr))
            elif char == ord('-') :
               incr/=10.0
               print('sensitivity = {}\n\r'.format(incr))
    finally:
        # shut down cleanly
        curses.nocbreak(); curses.echo()
        curses.endwin()
        obj.turnOff()

def testPins(pins) :
    '''
    pins may be:
        'all' - flash all GPIO pins (may conflict with HDMI display if in use)
        A single pin string - eg 'P8_09' (physical pin #9 on the P8 header)
        A list of pin strings
    '''
    if pins == 'all' :
        pins = ['P8_{}'.format(i) for i in range(3,46)] + [
                'P9_{}'.format(i) for i in range(11,19)+range(21,32)+range(41,43)]
    elif isinstance(pins, str) :
        pins = [pins]
    for p in pins :
        GPIO.setup(p,1)
    try:
        while True :
            for p in pins : GPIO.output(p,1)
            time.sleep(.2)
            for p in pins : GPIO.output(p,0)
            time.sleep(.2)
    except KeyboardInterrupt :
        print('Pin test done')


# Tests
if __name__ == "__main__" :
    intf = Interferometer()
    [mirror1, mirror2, mirror3, mirror4] = intf.mirrors

    arm1 = intf.arm1
    arm2 = intf.arm2

