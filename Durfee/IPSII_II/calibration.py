import numpy as np

# This code helps find the calibration numbers to move mirrors in sync
# The goal is that each arm of the intf should be able to change angle without
# changing the beam position *at the plane of the object*
#
# Unfortunately, the axis are not independent, which means a horizontal
# adjustment in one mirror requires horizontal and vertical adjustments in the
# second. However, this effect is small, so if the intf is set up symmetrically,
# and the object is about the same distance away as mirror 2 is from mirror 1,
# then the correction should be close to a factor of -2 in the same axis. i.e. if
# m1 is set to angle (a,b) then mirror two should be set to angle (-2a, -2b).
# This is a good starting point for what to expect.
#
# points are in this format:
# [ mirror1_position, mirror2_position]
# = [(h1,v1), (h2, v2)] (where h1, etc. absolute steppermotor positions, or angles)
#
# To find calibration, make beam small (with iris), put an iris directly in
# front of a detector at the object location, connect to o-scope. Move m1 (mirror1) to
# maximize signal. Reset home coordinates. Set m2 to some angle (towards edge of
# range. manualMove(m1) to get the beam back on center and maximze signal again.
# Copy coordinates from:
#
#   [m1,m2,m3,m4] = intf.mirrors
#   [(m.stepperH.position, m.stepperV.position) for m in [m1,m2]]
#
# Repeat for several angles. Paste points below in an array. Run array through
# code for linear regression fits. This will tell you the calibration values, as
# in if m1 is set to (a,b) the m2 should be set to
#        (a2, b2) = (h1*a+v1*b+c1, h2*a+v2*b+c2)
# The fit values should be close to (h1, v1, h2, v2) = (2, 0, 0, 2)
# The resulting beam angle is approximately (a2-a1, b2-b1)
#
# Then this whole process should be repeated for arm2 (m3 and m4)
#
# Note, the fit is done twice, once allowing for a fit offset, c, which should be
# close to 0 (if you started at the right position), and once without the offset
#
# From mathematica - if you want to go to intf angle to (th,tv)
# a = -((c1-th-c2 v1+tv v1+c1 v2-th v2)/(1+h1-h2 v1+v2+h1 v2))
# b = -((-c2-c2 h1+c1 h2-h2 th+tv+h1 tv)/(-1-h1+h2 v1-v2-h1 v2))
# then m1 = (a, b)
# and  m2 = (h1*a+v1*b, h2*a+v2*b) ~= (28th
# but beam angle is 2x mirror angle, and intf angle = arm1+arm2
# so for beam angle (th, tv) you should input (th/4, tv/4)

###Jarom's IPSII ONE setup
## Dec 17, 2018
#points_arm1 = [ [(70225, 70225), (-162690, -146538)],
#                [(46817, 70225), (-110020, -144431)],
#                [(23408, 23408), (-53137, -48221)]]
#
## Dec 18, 2018
#points_arm1 = [ [(0, 0), (0, 0)],
#               [(-234, 234), (0, 0)],
#               [(-21067, -22706), (46817, 46817)],
#               [(-22706, 22706), (46817, -46817)],
#               [(20131, 23174), (-46817, -46817)],
#               [(21535, -22238), (-46817, 46817)] ]
#
## Dec 19, 2018
#points_arm2 = [ [(0, 0), (0, 0)],
#               [(-20131, -23408), (46817, 46817)],
#               [(-22238, 22238), (46817, -46817)],
#               [(20599, 22238), (-46817, -46817)],
#               [(22472, -23408), (-46817, 46817)],
#               [(0, 0), (0, 0)]]
## Feb 27, 2018
#points_arm1 = [[(0,0), (0,0)],
#               [(23408, 0), (-50328, 234)],
#               [(-23408, 23408), (47753, -47753)],
#               [(23408, -23408), (-47753, 48689)],
#               [(-23408, -23408), (53371, 47753)],
#               [(23408, 23408), (-53137, -47285)],
#               [(0, 23408), (-2574, -47519)]
#               ]
#points_arm2 = [[(0, 0), (0, 0)],
#               [(0, 23408), (-2106, -49860)],
#               [(23408, 0), (-51264, -3745)],
#               [(23408, 23408), (-53371, -51499)],
#               [(-23408, -23408), (52201, 44008)],
#               [(-23408, 0), (51030, -3043)],
#               [(0, -23408), (1638, 45178)]
#               ]

### IPSII_II setup
# July 1, 2019
points_arm1 = [[(0,0), (0,0)],
               [(-17322, -18258), (35112, 35112)],
               [(16854, 17322), (-35112, -35112)],
               [(-29260, -30899), (60862, 60862)],
               [(29963, 31133), (-60862, -60862)],
               [(23408, 24579), (-49158, -49158)],
               
               ]
points_arm2 = [[(0,0), (0,0)],
               [(-24110, -24579), (49158, 49158)],
               [(-29494, -30665), (60862, 60862)],
               [(-16854, -17322), (35112, 35112)],
               [(18492, 18024), (-35112, -35112)],
               [(25515, 25281), (-49158, -49158)],
               [(31601, 31367), (-60862, -60862)],
               [(23174, -10299), (-44476, 21067)],##with these three lines, it was so much better
               [(56882, -22940), (-110020, 86611)],
               [(-8661, 29260), (18726, -11704)]
               ]
# July 2, 2019
points_arm1 += [[(-6554, 20833), (11704, -39794)],
               [(20599, -6086), (-39794, 11704)],
               [(-8895, 3511), (16386, -7022)]
               ]
# July ?, 2019
#points_arm1 = [[],
#               [],
#               [],
#               [],
#               [],
#               [],
#               []
#               ]
#points_arm2 = [[],
#               [],
#               [],
#               [],
#               [],
#               [],
#               []
#               ]

# Insert data set here
datasets = [np.vstack([np.hstack(p) for p in points_arm1]),
            np.vstack([np.hstack(p) for p in points_arm2])]
names = ['Arm1', 'Arm2']


# Perform regressions for each arm
for j in range(len(datasets)):
    p = datasets[j]
    print("Fitting {} 1st without offset, second with (better?)".format(names[j]))

    # Perform the regression both with an offset, and without
    # (If calibration was done properly, offset should be 0)
    for i in [0,1]:
        m1 = np.c_[p[:,0:2], i*np.ones(len(p))]
        m2H = p[:,2]
        m2V = p[:,3]
        h1, v1, c1 = np.linalg.lstsq(m1, m2H)[0]
        h2, v2, c2 = np.linalg.lstsq(m1, m2V)[0]
        fs = "{:+04.4f}"
        print('Horizontal m2 axis h1, v1, c1, : {0}, {0}, {0}'.format(fs).format(h1, v1, c1))
        print('Vertical m2 axis h2, v2, c2, : {0}, {0}, {0}'.format(fs).format(h2, v2, c2))
        fs = "{:+04.5f}"
        print('Calibration String : [{0},{0},{0},{0}]'.format(fs).format(h1,v1,h2,v2))
    print('\n')



