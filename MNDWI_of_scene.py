import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

import netCDF4 as nc
import time

import make_RGB


def mndwi_mask(aB3, aB6, aB11):
    """
	Input: three 2D arrays of the same size
	---
    Return an array that can be used to mask land pixels based on MNDWI index (Xu et al 2006)
	---
	Output: 2D array of MNDWI mask
    """

    # Covert the DN into reflectance values.
    aB3Reflectance = aB3
    aB6Reflectance = aB6

    # Apply the mask from band 11 to remove black pixels from the image.
    aB11Mask = np.ma.getmaskarray(aB11)
    aB3Reflectance = np.ma.array(aB3Reflectance, mask=aB11Mask)
    aB6Reflectance = np.ma.array(aB6Reflectance, mask=aB11Mask)

    # Calculate the MNDWI to find the water.
    aMNDWIMask = mndwi(aB3Reflectance, aB6Reflectance, fCutoff=0.8)
    return aMNDWIMask
	
def mndwi(aB3, aB6, fCutoff=0.8):
    """
	Input:
	---
    Modification of Normalized Difference Water Index.
    The B3 and B6 arrays need to be reflectance values.
    High values (over cutoff) are water, low values are land.
    The MNDWI is normalised using percentiles.
	---
	Output:
    """
	
    aMNDWI = (aB3 - aB6) / (aB3 + aB6)

    #max = np.amax(aMNDWI)
    #min = np.amin(aMNDWI)
	
    # Normalise so ranges using percentiles
    fPercentailMax = np.percentile(aMNDWI, 99.9)
    fPercentailMin = np.percentile(aMNDWI, 0.01)
    aMNDWI -= fPercentailMin
    aMNDWI /= fPercentailMax - fPercentailMin
    #aMNDWI = np.ma.masked_where(aMNDWI >= fPercentail, aMNDWI)

    return aMNDWI

def horizontal_samples(aMNDWIMask):
	"""
	Input:
	---
	kk
	---
	Output: 
	"""
	sample100 = aMNDWIMask[:][300]
	vector100 = np.where(sample100 > 0.8, 100, 0)
	sample_masked100 = np.ma.masked_where(vector100 < 100, vector100)

	sample200 = aMNDWIMask[:][200]
	vector200 = np.where(sample200 > 0.8, 200, 0)
	sample_masked200 = np.ma.masked_where(vector200 < 200, vector200)

	sample300 = aMNDWIMask[:][100]
	vector300 = np.where(sample300 > 0.8, 300, 0)
	sample_masked300 = np.ma.masked_where(vector300 < 300, vector300)

	sample150 = aMNDWIMask[:][250]
	vector150 = np.where(sample150 > 0.8, 150, 0)
	sample_masked150 = np.ma.masked_where(vector150 < 150, vector150)

	sample250 = aMNDWIMask[:][150]
	vector250 = np.where(sample250 > 0.8, 250, 0)
	sample_masked250 = np.ma.masked_where(vector250 < 250, vector250)
	
	return sample200, sample_masked100, sample_masked200, sample_masked300, sample_masked150, sample_masked250
	
# read in the netcdf file, file given as argument in the command line	
filename = sys.argv[1]	
dataset = nc.Dataset(filename, 'r+' , format="NETCDF4")

#pick your variables for the mask and for BT
refl_band3 = np.array(dataset.variables['reflectance_band3'])
#refl_band3 = np.rot90(refl_band3)
refl_band6 = np.array(dataset.variables['reflectance_band6'])
#refl_band6 = np.rot90(refl_band6)
BT11 = np.array(dataset.variables['BT_band11'])	

# Get the MNDWI
aMNDWIMask = mndwi_mask(refl_band3, refl_band6, BT11)
#aMNDWIMask = np.rot90(aMNDWIMask)

sample200, sample_masked100, sample_masked200, sample_masked300, sample_masked150, sample_masked250 = horizontal_samples(aMNDWIMask)

rgb = make_RGB.file_to_rgb(filename)

fig, ax = plt.subplots()
plt.title('Heysham RGB with detected samples of water pixels (red)')
ax.imshow(rgb, extent=[0, 400, 0, 400])
ax.plot(sample_masked100, c='r', linewidth=2.0)
ax.plot(sample_masked150, c='r', linewidth=2.0)
ax.plot(sample_masked200, c='r', linewidth=2.0)
ax.plot(sample_masked250, c='r', linewidth=2.0)
ax.plot(sample_masked300, c='r', linewidth=2.0)
plt.show()



plt.subplot(211)
plt.plot(sample200, label='MNDWI values')
plt.axhline(y=0.8, xmin=0, xmax=400, linewidth=2, linestyle='--' , color='red', label='0.8 threshold')
plt.axhline(y=0.9, xmin=0, xmax=400, linewidth=2, linestyle='--' , color='orange', label='0.9 threshold')
plt.xlabel('No. of pixels in the longitude' , fontsize=10)
plt.ylabel('MNDWI values' , fontsize=10)
plt.legend(loc='best')
#plt.legend(bbox_to_anchor=(1, 1) , loc=2, fontsize=10) # borderaxespad=0. ,
plt.title('MNDWI for central latitude across all longitude values using 99.9th percentile') # , fontsize=12)
# Adjust the subplot layout, because the logit one may take more space than usual
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout()
plt.subplot(223)
plt.imshow(aMNDWIMask)
plt.axhline(y=200, xmin=0, xmax=400, linewidth=2, color = 'k')
plt.colorbar()
plt.clim()
plt.title ('MNDWI (Xu et al, 2006) of the scene with cross scetion on central latitude') #('Reflectance band 3 - Green \n (0.53-0.59 micron)' , fontsize=12)
#plt.subplot(235)
#plt.imshow(refl_band6)
#plt.axhline(y=200, xmin=0, xmax=400, linewidth=2, color = 'k')
#plt.title('Reflectance band 6 - SWIR \n (1.57-1.65 micron)' , fontsize=12)
plt.subplot(224)
plt.imshow(rgb, extent=[0, 400, 0, 400])
#ax.plot(sample_masked100, c='r', linewidth=2.0)
#ax.plot(sample_masked150, c='r', linewidth=2.0)
plt.plot(sample_masked200, c='m', linewidth=2.0)
#ax.plot(sample_masked250, c='r', linewidth=2.0)
#ax.plot(sample_masked300, c='r', linewidth=2.0)
plt.title('RGB image with clear water pixels marked on the central latitude') # , fontsize=12)

plt.show()
#output_name = sys.argv[2]
#plt.savefig(output_name, bbox_inches='tight', pad_inches=0.5) 


