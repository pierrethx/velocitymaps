# Per Emission Line, Modify filename, vmin/vmax, cbar, savefig, zoom                                      

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import constants as c, units as u
from astropy import cosmology
from astropy.wcs import WCS
import seaborn as sns
from astropy.visualization.wcsaxes import SphericalCircle


sns.set_style("ticks", {"xtick.direction": u"in", "ytick.direction": u"in"})
fontsize = 14
plt.rc("font", **{"family": "sans-serif", "serif": ["Arial"]})
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)

#f606 = fits.open("J1059_F606W_sci.fits")
#f606 = fits.open("/Volumes/pr/scr3/cos/hst12533/images/cutouts/f625/23365+3604_625_credit_trim10.fits")
#f606 = fits.open("image.J024815-081723_icubes.wc.c5008_29.fits")  # has PHOTFLAM ADDED
#f606 = fits.open("image.J024815-081723_icubes.wc.c4686_13.fits")  # has PHOTFLAM ADDED
#f606 = fits.open("image.J024815-081723_icubes.wc.c3728_15.fits")  # has PHOTFLAM ADDED
#f606 = fits.open("O32_bin_new.fits")  # has PHOTFLAM ADDED & edited

# O/H
#f606 = fits.open("J024815-081723_icubes.wc.c_OH.fits")
f606 = fits.open('/Volumes/TOSHIBA EXT/MARTINLAB/target10/z_image.J024815-081723_icubes.wc.c4861_assigned.fits')  


# A_V 
#f606 = fits.open("J024815-081723_icubes.wc.c_A_V.fits")
#f606 = fits.open("J024815-081723_icubes.wc.c_A_V_negative_blocked.fits")
#f606 = fits.open("J024815-081723_icubes.wc.c_A_V_chi2_blocked.fits")
#f606 = fits.open("J024815-081723_A_V_caseB_2.fits")

# Te
#f606 = fits.open("J024815-081723_Te_caseB_2.fits")

#f606 = fits.open("/Users/pengxuyun/Desktop/Prof.Martin/KCWI_Data_Cubes/Example_Galaxy/J0248-0817/J024815-081723_icubes.wc.c_OH.fits")
#f606 = fits.open("image.J024815-081723_icubes.wc.c4862_29.fits")   # has PHOTFLAM ADDED

#f606 = fits.open('/Volumes/pr/scr8/kcwi/data/addcubes/binned/j0248-0817/target3/block_clmbin.J024815-081723_icubes.wc.c4686_wit_sig.fits')  # has PHOTFLAM ADDED



wcs606 = WCS(f606[0].header)
#im606 = f606[0].data * f606[0].header["PHOTFLAM"] * u.erg / u.s / u.cm ** 2 / u.angstrom
im606 = f606[0].data 


#fig = plt.figure(num=1, figsize=(8, 7))   # width, height in inches
fig = plt.figure(num=1, figsize=(10, 7))   # width, height in inches
plt.clf()    # clears the entire current figure

ax = plt.subplot(projection=wcs606)
# No colorbar
#ax.imshow(im606.value, origin="lower", vmin=1e-20, vmax=6e-18, cmap="magma")
# With colorbar
#flux = ax.imshow(im606.value, origin="lower", vmin=1e-20, vmax=6e-18, cmap="magma")     # O3
#flux = ax.imshow(im606.value, origin="lower", vmin=1e-20, vmax=1.79e-18, cmap="magma")  #He II
#flux = ax.imshow(im606.value, origin="lower", vmin=1e-20, vmax=6e-18, cmap="magma")      # O2
flux = ax.imshow(im606, origin="lower", cmap="plasma")     
#flux = ax.imshow(im606.value, origin="lower", vmin=1e-20, vmax=6e-18, cmap="magma")      # Hb
#flux = ax.imshow(im606.value, origin="lower", vmin=0, vmax=6e-18, cmap="magma")  #He II bin

cbar = fig.colorbar(flux)
cbar.ax.get_yaxis() .labelpad=15
#cbar.ax.set_ylabel('SB ([O III] 5007) (ergs/s/cm2/pix)', rotation=90, fontsize=18)
#cbar.ax.set_ylabel('SB (He II 4686) (ergs/s/cm2/pix)', rotation=90, fontsize=18)  #HeII
#cbar.ax.set_ylabel('SB ([O II]) (ergs/s/cm2/pix)', rotation=90, fontsize=18)       #O2
cbar.ax.set_ylabel('12 + Log(O/H)', rotation=90, fontsize=18)   
#cbar.ax.set_ylabel('O32', rotation=90, fontsize=18)            
#cbar.ax.set_ylabel('$A_v$', rotation=90, fontsize=18)  
#cbar.ax.set_ylabel('SB (Hb) (ergs/s/cm2/pix)', rotation=90, fontsize=18)       # Hb

## Set up the major ticks.
## Use different colors to identify which axis is which.
ax.coords['ra'].set_ticks(color='black')
#ax.coords['dec'].set_ticks(color='r')  # just to verify axis
ax.coords['dec'].set_ticks(color='black')  
ax.tick_params(axis='both', which='major', length=10)
#clm:  we are working with the 'ax.coords' object. Draw grid ...
ax.coords.grid(color='black', linestyle='solid', alpha=0.5)

# these 2 lines are crucial; specify where to show labels:
# l = left; b = bottom; recall ax.coords['ra'] is the vertical axis
ax.coords['dec'].set_ticklabel_position('l')
ax.coords['ra'].set_ticklabel_position('b')
ax.coords['ra'].set_axislabel('RA', fontsize=18)
ax.coords['dec'].set_axislabel('DEC', fontsize=18)

# Set the title of the axes object
ax.set_title('J0248-0817', loc='center', fontsize=18)


#lon = ax.coords["ra"]
#lat = ax.coords["dec"]
#lon.set_major_formatter("dd:mm:ss")
#lat.set_major_formatter("dd:mm:ss")
#lat.set_axislabel("Declination (J2000)", fontsize=fontsize, minpad=-0.4)
#lon.set_axislabel("Right Ascension (J2000)", fontsize=fontsize)

# zoom in on the part of the image we want
#plt.gca().axis([650, 930, 650, 930])
#plt.gca().axis([50, 110, 42, 102])  # For He II


# add a circle at the center; ds9[x,y] ==> [y-1,x-1]                                                
#ax.scatter([58.95], [61.978], s=400, edgecolor='green', facecolor='none')                          

#Fiber                                                                                              
# add a circle at the center; ds9[x,y] ==> [y-1,x-1]                                                
#ax.scatter([58.95], [61.978], s=400, edgecolor='green', facecolor='none')                          

#Fiber                                                                                              
#r = SphericalCircle((42.066387 * u.deg, -8.2878333 * u.deg), 0.0001388888888888889 * u.degree, edgecolor='green', facecolor='none', transform = ax.get_transform('fk5'))
#ax.add_patch(r)

"""
overlay = ax.get_coords_overlay('fk5')   # WCS Coords method
overlay.grid(color='white', ls='dotted')
overlay['ra'].set_ticklabel_position('b')
overlay['dec'].set_ticklabel_position('l')
overlay['ra'].set_axislabel('New RA')
overlay['dec'].set_axislabel('New DEC')
"""

#plt.savefig("J0248-0817_5007.png")
#plt.savefig("J0248-0817_4686.png")
#plt.savefig("J0248-0817_3727.png")
#plt.savefig("J0248-0817_Hb.png")
#plt.savefig("J0248-0817_O32.png")
#plt.savefig("J0248-0817_O32_binned.png")
#plt.savefig("J0248-0817_4686_bin3.png")

#plt.show()
plt.show()
