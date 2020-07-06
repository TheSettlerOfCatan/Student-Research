import numpy as np
import maplotlib.image as mpimg

#############################################################################################################
#Forward Fourier transform function with shadow mask inside the integrand 
#############################################################################################################

def transform_with_shadow(object_arr, reflectivity_arr, folder, size, num_digits):
    Object = object_arr
    Reflectivity = reflectivity_arr
    
    Nrow = np.shape(Object)[0] # num rows
    Ncol = np.shape(Object)[1] # num columns

    xr = range(Nrow)
    xc = range(Ncol)
    XC,XR = np.meshgrid(xc,xr)
    integrand = np.empty(np.shape(Object), dtype=np.complex)
    IPSII = np.empty(np.shape(Object), dtype=np.complex)
    image_filepath = "/home/carter/Documents/Research/simple cylinder 2/"+folder+"/shadows"
    
    for m in range(size):
        for n in range(size):
            
            shadow = mpimg.imread(image_filepath+'/m'+str(m+1).zfill(num_digits)+"/test"+str(m+1).zfill(num_digits)+str(n+1).zfill(num_digits)+".png")[:,:,0]
            
            integrand = Reflectivity*np.exp((-1j)*2*np.pi*(m*XR/Nrow + n*XC/Ncol))*shadow
            IPSII[m,n] = sum(sum(integrand))
            
## theta = np.arcsin(l*np.sqrt(((m*deltakm)**2)+((n*deltakn)**2))/2)
## phi = np.arctan()
            
    return IPSII

    
#############################################################################################################
#Simulating IPSII with the forward transform, then taking the inverse transform for the resulting image
#############################################################################################################
 
IPSII128random = transform_with_shadow(gimg128, randomgimg128, "128x128", 128, 3)
IPSII256random = transform_with_shadow(gimg256, randomgimg256, "256x256", 256, 3)
IPSII512random = transform_with_shadow(gimg512, randomgimg512, "512x512", 512, 3)

plt.imshow(np.fft.ifft2(IPSII128random).real)
plt.imshow(np.fft.ifft2(IPSII256random).real)
plt.imshow(np.fft.ifft2(IPSII512random).real)


