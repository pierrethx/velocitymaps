import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename
from astropy.io import fits
from astropy import wcs
from matplotlib import colors
from matplotlib.widgets import Slider 

maskk=True

root=tkinter.Tk()
root.withdraw()
placeholder=askopenfilename(message="Select 2d fits file to plot")
while(placeholder!=''):
    
    root.update()
    root.destroy()

    with fits.open(placeholder,checksum=True) as hdul:
        signal0=np.flipud(hdul[0].data)
        wcsx=hdul[0].header
    
    if maskk:
        root=tkinter.Tk()
        root.withdraw()
        pj=askopenfilename(message="Select 2d fits file to serve as mask")
        with fits.open(pj,checksum=True) as hdul:
            mask=np.flipud(hdul[0].data)
            wcsx2=hdul[0].header
        
        root.update()
        root.destroy()

        signal=np.ma.masked_where(mask==0,signal0)
    else:
        signal=signal0
    
    #signal=np.ma.masked_outside(signal,-1,10000)
    fl=signal.flatten()

    fig,ax=plt.subplots(1,2)
    
    q=ax[0].imshow(signal,cmap="cubehelix",vmin=-0.5,vmax=0.5)
    qc=plt.colorbar(q)
    q2=ax[1].hist(fl,100)
    plt.subplots_adjust(left=0.05,bottom=0.25)

    def update(val):
        maxx=smax.val
        minx=smin.val
        #ax.imshow(signal,cmap="cubehelix",vmin=minx,vmax=maxx)
        qc.mappable.set_clim(vmin=minx,vmax=maxx)
        ax[1].clear()
        q2=ax[1].hist(fl,100,(minx,maxx))
        plt.show()

    maxax=plt.axes([0.25,0.15,0.65,0.03],facecolor="beige")
    smax=Slider(maxax,'Max val',0,0.5,valinit=0.5,valstep=0.001,orientation="horizontal")
    
    minax=plt.axes([0.25,0.10,0.65,0.03],facecolor="beige")
    smin=Slider(minax,'Min val',-0.5,0,valinit=-0.5,valstep=0.001,orientation="horizontal")
    smax.on_changed(update)
    smin.on_changed(update)
    plt.show()

    root=tkinter.Tk()
    root.withdraw()
    placeholder=askopenfilename(message="Select 2d fits file to plot")
print("Bye bye!")
