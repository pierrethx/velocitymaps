from astropy.io.fits import file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import cm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Table
from numpy.core.numeric import zeros_like
from scipy.optimize import curve_fit
from IPython import embed   # add embed()
import time

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

binning="target5/"
lookat="vel5007_bin.fits"
saveas="_vel5007_fig.png"
bin_file_list=[]
dest_list=[]
extlist=[]
name=[]
#j0823
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j0823+0313_17frames_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/j0823/"+binning)
extlist.append(0.0002)
name.append("j0823")

#j1238
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1238+1009_main_icubes5006_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/j1238/"+binning)
extlist.append(0.0001)
name.append("j1238")

#j0248
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.J024815-081723_icubes.wc.c3728_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/j0248/"+binning)
name.append("j0248")
extlist.append(0.0001)

#j1044
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1044+0353_addALL_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/j1044/"+binning)
name.append("j1044")
extlist.append(0.0001)



for fil in range(len(dest_list)):
    for maskky in [True,False]:
        with fits.open(dest_list[fil]+lookat,checksum=True) as hdul:
            signal=hdul[0].data
        with fits.open(bin_file_list[fil],checksum=True) as hdul:
            mask=hdul[0].data
            wcsx=WCS(hdul[0].header)
        #lower=input("Input lower bound: ")
        #upper=input("Input upper bound: ")
        fig=plt.figure()
        
        ax=plt.axes(projection=wcsx)
        
        signalm=np.ma.masked_where(mask==0,signal)
        if maskky:
            signalm=np.ma.masked_where(np.abs(signalm)>extlist[fil],signalm)
        g=ax.imshow(signalm,origin='lower',vmin=extlist[fil],vmax=-extlist[fil],cmap='seismic')
        plt.grid(color='black',ls='solid')
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        
        plt.colorbar(g)
        
        print("before")
        if maskky:
            plt.savefig("/Volumes/TOSHIBA EXT/MARTINLAB/pics/m_"+name[fil]+saveas)
        else:
            plt.savefig("/Volumes/TOSHIBA EXT/MARTINLAB/pics/"+name[fil]+saveas)
        print("after")
        plt.close(fig)