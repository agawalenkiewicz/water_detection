import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys


def refl_to_DN(reflectance_data):
    #https://landsat.usgs.gov/landsat-8-l8-data-users-handbook-section-5
	"""
	Input: 2D array of reflectance values for each pixel
	---
	Convert the TOA Reflectance to DN as described in primer Landsat User's guide document.
	M_rho = Band-specific multiplicative rescaling factor from the metadata
        (REFLECTANCE_MULT_BAND_x, where x is the band number)
	A_rho = Band-specific additive rescaling factor from the metadata
        (REFLECTANCE_ADD_BAND_x, where x is the band number)
	---
	Output: 2D array of DN values of each pixel
	"""

	M_rho = 2.0E-05
	A_rho = -0.1
	
	image_data = np.ma.copy(reflectance_data)
	image_data -= A_rho
	image_data /= M_rho
	
	return image_data

def make_rgb(ch1,ch2,ch3):
    """
	Input: three 2D arrays of DN values (DN_ digital number 0-256)
	---
    Make RGB image from Red, Green, Blue.
    Note that the information from each channel has to be in DN.
    The channels need to come in order R-G-B
    ch1 - red
    ch2 - green
    ch3 - blue
	---
	Output: single 2D array of RGB image
    """
    f = np.ma.zeros([ch1.shape[0],ch1.shape[1],3])
    f1max = np.nanmax(ch1)
    f1min = np.nanmin(ch1)
    f2max = np.nanmax(ch2)
    f2min = np.nanmin(ch2)
    f3max = np.nanmax(ch3)
    f3min = np.nanmin(ch3)

    if (f1max > 200):
        f[:,:,0] = abs((ch1-f1min)/(f1max-f1min) - 1)
    else:
        f[:,:,0] = (ch1-f1min)/(f1max-f1min)
    if (f2max > 200):
        f[:,:,1] = abs((ch2-f2min)/(f2max-f2min) - 1)
    else:
        f[:,:,1] = (ch2-f2min)/(f2max-f2min)
    if (f3max > 200):
        f[:,:,2] = abs((ch3-f3min)/(f3max-f3min)-1)
    else:
        f[:,:,2] = (ch3-f3min)/(f3max-f3min)

    return f



def file_to_rgb(filename):
	"""
	#Wrapper function.
	Input: netcdf file
	---
	read in a netcdf file
	if needed, convert from reflectance values to DN
	combine DN from three channels to obtain RGB image
	---
	Output: 2D RGB data
	"""
	# read in the netcdf file, file given as argument in the command line		
	#dataset = nc.Dataset(filename)
	dataset = filename #nc.Dataset(filename, 'r+' , format="NETCDF4")

	#pick your variables for the mask and for BT
	blue = np.array(dataset.variables['reflectance_band2'])
	#blue = np.rot90(blue)
	green = np.array(dataset.variables['reflectance_band3'])
	#green = np.rot90(green)
	red = np.array(dataset.variables['reflectance_band4'])
	#red = np.rot90(red)

	ch1 = refl_to_DN(red)
	ch2 = refl_to_DN(green) 
	ch3 = refl_to_DN(blue) 
	
	rgb = make_rgb(ch1,ch2,ch3) #(red, green, blue)
	return rgb

#f = file_to_rgb(sys.argv[1])
#plt.imshow(f,interpolation='nearest')
#plt.show()