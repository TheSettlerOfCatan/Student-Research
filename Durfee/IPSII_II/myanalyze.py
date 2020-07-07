import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import hilbert as hilbert
from scipy import ndimage

folder = sys.argv[1]
params = json.load(open('dataruns/'+folder+'params_set.json'))
N = params['x_lastN']
x_FOV = params['x_FOV']
y_FOV = params['y_FOV']
all_the_data = np.load(folder+'all_the_data.npy')
begin_good_data = 25
end_good_data = 425

def update_progress(workdone):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone*100), end="", flush=True)

data = all_the_data[:,:,:,begin_good_data:end_good_data]
C = np.empty((2*N+1,2*N+1),dtype=complex)

i = 0
for x in range(-N,N+1):
    for y in range(-N,N+1):
        ref = data[x,y,0,:]               
        ref -= np.mean(ref) #remove the DC offset
        obj = data[x,y,2,:]
        #print(obj.shape,ref.shape)
        qref = np.imag(hilbert(ref)) #the phase-shifted reference

        A = np.mean(ref*obj) #the in phase component
        B = np.mean(qref*obj) #the quadrature component
        C[x,y] = A + 1j*B

        i += 1
        update_progress(i/((2*N+1)**2))

#np.save(folder+'Cdata',C)


kspace = np.fft.fftshift(np.log10(np.abs(C)+1e-16))
np.save(folder+'kspace',kspace)
plt.imsave(folder+'kspace.png',kspace,cmap='inferno')



object_image = np.abs((np.fft.ifftshift(np.fft.ifft2(C))))
np.save(folder+'object',object_image)
object_max = np.unravel_index(object_image.argmax(), object_image.shape)
if (object_max[0] in range(N-3,N+3)) and (object_max[1] in range(N-3,N+3)):
    object_image[object_max] = 0
plt.imsave(folder+'object.png',object_image,cmap='bone')


plt.subplots()
plt.subplot(2,2,1)
plt.imshow(kspace,cmap='inferno')

plt.subplot(2,2,2)
plt.imshow(np.angle(C))

plt.subplot(2,2,3)
plt.imshow(object_image,cmap='bone')

img = np.fft.fftshift(np.fft.ifft2(C))
plt.subplot(2,2,4)
plt.plot(np.angle(img))
plt.show()



#plt.figure(1, dpi=100, figsize=(6,10))
#plt.subplot(2,1,1)
#plt.imshow(kvalues_image, cmap='inferno', aspect='equal', interpolation='nearest')
#plt.text(.4, 1, 'kvalues', transform=plt.gca().transAxes, backgroundcolor='white')

#plt.subplot(2,1,2)
#plt.imshow(object_image, cmap='bone', extent=(0,x_FOV,0,y_FOV), aspect='equal', interpolation='nearest')
#plt.text(0, 1, 'object image', transform=plt.gca().transAxes, backgroundcolor='white')
#plt.xlabel('mm')
#plt.ylabel('mm')
#plt.savefig(folder+'myanalyze results')
#plt.show()
#plt.close('all')
print()





