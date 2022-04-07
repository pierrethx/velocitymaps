from astropy.io.fits import file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

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

def makesubfolder(sourcedir,subfolder):
    try:
        os.mkdir(sourcedir+"/"+subfolder)
    except:
        pass
        print("already exists")
    return sourcedir+"/"+subfolder

lookat="vel_bin.fits"
lookat2="vele_bin.fits"
saveas="_vel_fig.png"

subby="nebulae3/"
bin_file_list=[]
dest_list=[]
obj_list=[]
ext=40
cut=2.5
name=[]
for binning in ["target5/","target10/"]:
    for wl in ["4861","4958","5006"]:
        
        #j0823
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j0823+0313_17frames_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0823/"+wl+'_'+binning)
        obj_list.append("j0823+0313")
        name.append("j0823_"+wl+"_"+binning[:-1])

        #j1238
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1238+1009_main_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1238/"+wl+'_'+binning)
        obj_list.append("j1238+1009")
        name.append("j1238_"+wl+"_"+binning[:-1])

        #j0248
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.J024815-081723_icubes.wc.c'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0248/"+wl+'_'+binning)
        name.append("j0248_"+wl+"_"+binning[:-1])
        obj_list.append("J024815-081723")

        #j1044
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1044+0353_addALL_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1044/"+wl+'_'+binning)
        name.append("j1044_"+wl+"_"+binning[:-1])
        obj_list.append("j1044+0353")

        #j0944
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'/z_image.j0944-0039_addALL_1200_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0944/"+wl+'_'+binning)
        name.append("j0944_"+wl+"_"+binning[:-1])
        obj_list.append("j0944-0039")

        #j1418
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1418+2101_add1200_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1418/"+wl+'_'+binning)
        name.append("j1418_"+wl+"_"+binning[:-1])
        obj_list.append("j1418+2101")
        
        #j0837
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j0837+5138_all1200_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j0837/"+wl+'_'+binning)
        name.append("j0837_"+wl+"_"+binning[:-1])
        obj_list.append("j0837+5138")

        #j1016
        bin_file_list.append('/Volumes/TOSHIBA EXT/MARTINLAB/'+binning+'z_image.j1016+3754_addALL1200_icubes'+wl+'_assigned.fits')
        dest_list.append("/Volumes/TOSHIBA EXT/MARTINLAB/"+subby+"j1016/"+wl+'_'+binning)
        name.append("j1016_"+wl+"_"+binning[:-1])
        obj_list.append("j1016+3754")



for fil in range(len(dest_list)):
    with fits.open(dest_list[fil]+lookat,checksum=True) as hdul:
        signal=hdul[0].data
    with fits.open(dest_list[fil]+lookat2,checksum=True) as hdul:
        error=hdul[0].data
    with fits.open(bin_file_list[fil],checksum=True) as hdul:
        mask=hdul[0].data
        wcsx=WCS(hdul[0].header)
    dest=makesubfolder("/Volumes/TOSHIBA EXT/MARTINLAB/pics/","Dec24/")
    for maskky in [True,False]:
        #lower=input("Input lower bound: ")
        #upper=input("Input upper bound: ")
        fig=plt.figure()
        
        ax=plt.axes(projection=wcsx)
        if maskky:
            signalm=np.ma.masked_where(np.abs(signal/error)<cut,signal)
            signalm=299792*np.ma.masked_where(mask==0,signalm)
        else:
            signalm=299792*np.ma.masked_where(mask==0,signal)
        
        g=ax.imshow(signalm,origin='lower',vmin=-ext,vmax=ext,cmap='seismic')
        #ax.set_facecolor('black')
        plt.grid(color='black',ls='solid')
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        ax.set_title(obj_list[fil])
        
        if maskky:
            plt.colorbar(g,label="SNR")
            #ax.contour(signalm,[-5,-2.5,0,2.5,5],colors="green",linewidths=3,origin='lower')
        else:
            plt.colorbar(g,label="specific velocity (km/s)")
        
        subf="Dec24/"
        print("before")
        
        if maskky:
            plt.savefig(dest+"m_"+name[fil]+saveas)
        else:
            plt.savefig(dest+name[fil]+saveas)
        #plt.show()
        print("after")
        plt.close(fig)

    fig=plt.figure()
    
    ax=plt.axes(projection=wcsx)
    signalm=signal/error
    #signalm=np.ma.masked_where(np.abs(signal/error)<cut,signal)
    
    g=ax.imshow(signalm,origin='lower',vmin=-10,vmax=10,cmap='seismic')
    #ax.set_facecolor('black')
    plt.grid(color='black',ls='solid')
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title(obj_list[fil])
    
    plt.colorbar(g,label="SNR")
    #ax.contour(signalm,[-5,-2.5,0,2.5,5],colors="green",linewidths=3,origin='lower')
    
    plt.savefig(dest+"SNR_"+name[fil]+saveas)
    #plt.show()

    plt.close(fig)