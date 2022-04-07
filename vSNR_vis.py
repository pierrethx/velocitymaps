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
lookat="vel_bin.fits"
lookat2="vele_bin.fits"
saveas="_vel_fig.png"
subby="nebulae2/"
bin_file_list=[]
dest_list=[]
ext=40
name=[]
#j0823
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j0823+0313_17frames_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0823/"+binning)

name.append("j0823")

#j1238
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1238+1009_main_icubes5006_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1238/"+binning)

name.append("j1238")

#j0248
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.J024815-081723_icubes.wc.c3728_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0248/"+binning)
name.append("j0248")


#j1044
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1044+0353_addALL_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1044/"+binning)
name.append("j1044")

#j0944
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'/z_image.j0944-0039_addALL_1200_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0944/"+binning)
name.append("j0944")

#j1418
bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1418+2101_add1200_icubes3727_assigned.fits')
dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1418/"+binning)
name.append("j1418")




for fil in range(len(dest_list)):
    for cut in [1,2,3,4,5]:
        with fits.open(dest_list[fil]+lookat,checksum=True) as hdul:
            signal=hdul[0].data
        with fits.open(dest_list[fil]+lookat2,checksum=True) as hdul:
            error=hdul[0].data
        with fits.open(bin_file_list[fil],checksum=True) as hdul:
            mask=hdul[0].data
            wcsx=WCS(hdul[0].header)
        #lower=input("Input lower bound: ")
        #upper=input("Input upper bound: ")
        fig=plt.figure(cut)
        
        ax=plt.axes(projection=wcsx)

        SNR=np.abs(signal/error)
        SNRm=np.ma.masked_where(np.isnan(SNR),SNR)
        SNRm=np.ma.masked_where(mask==0,SNRm)
        
        signalm=np.ma.masked_where(SNRm<cut,signal)
        xx,yy=np.meshgrid(range(len(SNRm[0])),range(len(SNRm)))
        g=ax.imshow(signalm,origin='lower',vmin=-0.0002,vmax=0.0002,cmap='seismic')
        plt.grid(color='black',ls='solid')
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        
        plt.colorbar(g,label="log(SNR)")
        '''
        fig2,ax2=plt.subplots()
        SNRl=SNRm[:]
        ax2.hist(SNRl)
        ax2.set_xscale('log')
        '''
    plt.show()

        
        
    '''
    subf="Nov19/"
    print("before")
    if maskky:
        plt.savefig("/Volumes/TOSHIBA EXT/MARTINLAB/pics/"+subf+"m_"+name[fil]+saveas)
    else:
        plt.savefig("/Volumes/TOSHIBA EXT/MARTINLAB/pics/"+subf+name[fil]+saveas)
    print("after")
    plt.close(fig)
    '''