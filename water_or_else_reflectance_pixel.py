import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
from scipy import interpolate


def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - np.float(dd))).argmin()
   return geo_idx

   
#water_lat = sys.argv[1]
#water_lon = sys.argv[2]
#else_lat = sys.argv[3]
#else_lon = sys.argv[4]
water_lon_idx = sys.argv[1] #x
water_lat_idx = sys.argv[2] #y
else_lon_idx = sys.argv[3]
else_lat_idx = sys.argv[4]
place = sys.argv[5]
water_reflectance_b3 = []
water_reflectance_b6 = []
else_reflectance_b3 = []
else_reflectance_b6 = []


filename = ['LC82000252014034LGN01/scenes/LC08_L1TP_200025_20140203_20170426_01_T1.nc']
for element in filename:
	element_path = os.path.join('/glusterfs/surft/users/mp877190/data/datastore/EE/LANDSAT_8_C1/dungeness_checked', element)
	nc_file = nc.Dataset(element_path)
	"""
	print nc_file.variables.keys()
	lats = nc_file.variables['latitude'][:,0]
	lons = nc_file.variables['longitude'][0,:]

	water_lat_idx = geo_idx(water_lat, lats)
	water_lon_idx = geo_idx(water_lon, lons)
	else_lat_idx = geo_idx(else_lat, lats)
	else_lon_idx = geo_idx(else_lon, lons)
	"""
	
	water_reflectance_b3.append(nc_file.variables['reflectance_band3'][water_lon_idx, water_lat_idx])
	water_reflectance_b6.append(nc_file.variables['reflectance_band6'][water_lon_idx, water_lat_idx])
	else_reflectance_b3.append(nc_file.variables['reflectance_band3'][else_lon_idx, else_lat_idx])
	else_reflectance_b6.append(nc_file.variables['reflectance_band6'][else_lon_idx, else_lat_idx])
	
	print ("water refl b3" , water_reflectance_b3)
	print ("water refl b6" , water_reflectance_b6)
	print ("else refl b3" , else_reflectance_b3)
	print ("else refl b6" , else_reflectance_b6)




	
I_water = (np.array(water_reflectance_b3) - np.array(water_reflectance_b6)) / (np.array(water_reflectance_b3) + np.array(water_reflectance_b6))
I_else= (np.array(else_reflectance_b3) - np.array(else_reflectance_b6)) / (np.array(else_reflectance_b3) + np.array(else_reflectance_b6))

x=np.array([0,100])
print(x.shape)
f=np.array([I_else, I_water])
f=f.flatten()
print(f.shape)
g=np.array([0.8, 0.8])

fig = plt.figure()
ax1 =fig.add_subplot(211)
plt.plot(x,[else_reflectance_b3, water_reflectance_b3], label='Reflectance band 3')
plt.plot(x, [else_reflectance_b6, water_reflectance_b6], label='Reflectance band 6')
# Put a legend to the right of the current axis
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Band 3 and 6 reflectances')
plt.xlabel('% fraction of water')
plt.title('Land pixel (1st) vs. water pixel (2nd)')

ax2 = fig.add_subplot(212)
plt.plot(x,f, label='MNDWI for 2 chosen pixels')
plt.plot(x,g, label='Threshold value')
# Put a legend to the right of the current axis
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('MNDWI interpolated for those 2 pixels')

interp = interpolate.interp1d(f, x) # Create your function with your q (input) and z (output)
result = interp(0.8) # This is checking the value of z when q=0.8
print(result)

plt.plot([interp(0.8), interp(0.8)], [-0.2, 0.8], 'r--')
ax2.text(90, -0.2, '%f %%' %result, style='italic', fontsize=8, bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})


plt.suptitle(place, fontsize=20)
plt.tight_layout()
plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
#plt.show()
plt.savefig("land_vs_water.png",bbox_inches='tight')


