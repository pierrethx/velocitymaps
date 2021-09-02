# clm:  4/22/2021
#       6/22/2021 -- flag whether or not you are using  Ar4 assigned map; otherwise assume the O2 assigned map
# v2 - adds the radial plot of O32 values
# Package to extract spectra from a pixel list, such as those defined by binning a cube.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from IPython import embed   # add embed()
import time



"""
Notes:
      Set 'verbose = False' for production runs.

      m is the bin ID [nonconsecutive integers]
      nm is the index in binlist [0:nspec]

      Specify files and redshift in Main Program below.
      Move bin*_mask.fits
           spec1d*[F,E].fits
           O32_bin.fits
      to .../J####/

      Testing:
      Two places to test with 'verbose = True':
          Main Program
          fit_line()
      Inspect arrays o32 and ratio / ratio_err
      In ds9 inspect O32_bin.fits, bin#_mask.fits, bin_region_test_cube.fits

Action Items:
      You should combine all the mask files into one fits file.
      Setting the LDL flag for O2 in both fit_line and main is awkward.

"""
start=time.time()
func = lambda x, a, xcen, sigma:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2))  # Area = 

# a: amp_3729 and b = amp_3726
func2 = lambda x, a, xcen, sigma, b:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) + b * np.exp(-(x-[xcen-2.78999])**2 / (2 * sigma ** 2))   # Blend of [O II] 3726.03, 3728.82
#func_ldl = lambda x, a, xcen, sigma:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) + 2. * a / 3. * np.exp(-(x-[xcen-2.78999])**2 / (2 * sigma ** 2))   # 3726/3728 = 2/3, the LDL
func_ldl = lambda x, a, xcen, sigma:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) +  a / 1.47 * np.exp(-(x-[xcen-2.78999])**2 / (2 * sigma ** 2))   # PyNeb Max 3729/3726 = 1.47 in LDL
func_hdl = lambda x, a, xcen, sigma:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) +  a / 0.29 * np.exp(-(x-[xcen-2.78999])**2 / (2 * sigma ** 2))   # PyNeb Max 3729/3726 = 0.29 in HDL

# [ArIV] + He I blend (5 parameters; xcen is 4711 A, the higher J upper level)
# NOTE:  The wavlength difference is 28.85 in air; but this SHOULD BE REVISED TO 28.86 for vacuum
func3 = lambda x, a, xcen, sigma, b, c:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) + b * np.exp(-(x-[xcen + (4740.20-4711.35)])**2 / (2 * sigma ** 2)) + c * np.exp(-(x-[xcen + (4713.1392-4711.35)])**2 / (2 * sigma ** 2)) 

func3_ldl = lambda x, a, xcen, sigma, c:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) + a / 1.37 * np.exp(-(x-[xcen + (4740.20-4711.35)])**2 / (2 * sigma ** 2)) + c * np.exp(-(x-[xcen + (4713.1392-4711.35)])**2 / (2 * sigma ** 2)) 

func3_hdl = lambda x, a, xcen, sigma, c:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2)) + a / 0.76 * np.exp(-(x-[xcen + (4740.20-4711.35)])**2 / (2 * sigma ** 2)) + c * np.exp(-(x-[xcen + (4713.1392-4711.35)])**2 / (2 * sigma ** 2)) 


def show_bins(binmap):  
    """
    Input:  2D array [y,x]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

#    ax = plt.imshow(bin,cmap=cm.flag,origin='lower', vmin=bin.min(), vmax=bin.max())
    ax = plt.imshow(binmap,cmap=cm.flag,origin='lower', vmin=bin.min(), vmax=bin.max())
    cbar = fig.colorbar(ax)
    cbar.ax.get_yaxis() .labelpad=15
    cbar.ax.set_ylabel('Bin ID', rotation=270)

    plt.xlabel('X Image')
    plt.ylabel('Y Image')
    fig.show()

def reverseassign(map):
    m=int(np.nanmax(map))+1
    binlist=[]*m
    for y in len(map):
        for x in len(map[0]):
            binlist[int(map[y,x])]=(y,x)
    return binlist


def spec_extract(cube, vcube, mask):
    """
    Extract a spectrum from the cube.  Include spaxels not masked.

    Input:  cube (wavelength, y, x)  flux/pix
            vcube (variance)
            mask (y, x)
    Output:
            wave:  1D array
            flux:  1D array (per A)
            err:   1D array (per A)
    """

    nz = cube[:,0,0].size
    ny = cube[0,:,0].size
    nx = cube[0,0,:].size

    #############################
    # Select method of extraction
    #############################
    ifancy = True

    if ifancy:
        a=time.time()
        if mask.sum()>1:
            
            spec=np.einsum('ij,kij->k',mask,cube)
            espec=np.einsum('ij,kij->k',mask,vcube)
            print("einsum time",time.time()-a,"binsize",mask.sum())
        else:
            # Use list comprehension
            coords = [(j,i) for j in range(mask[:,0].size) for i in range(mask[0,:].size) if mask[j,i] !=0]  # select spaxels
            speclist =  [cube[:,j,i] for (j,i) in coords]   #  returns a list of 1D arrays in k-space
            allspec = np.array(speclist)                    #  2D array in (spaxel number, spectrallength)or (len(coords), nz)
            spec = np.sum(allspec, axis=0)                  #  smash the spaxels into a 1D spectrum
            especlist =  [vcube[:,j,i] for (j,i) in coords]   
            eallspec= np.array(especlist)                     
            espec = np.sum(eallspec, axis=0)
            print("comprehension time",time.time()-a)  
    else:
        # use loop over all spaxels for complete flexibility
        spec = np.zeros(nz)
        espec = np.zeros(nz)
        for i in range(nx):
            for j in range(ny):
                if mask[j,i] != 0:
                    spec = spec + cube[:,j,i]
                    espec = espec + vcube[:,j,i]


    espec[np.isnan(espec)] = 9e9  # Flag Nans with 9e9
    espec[espec<0] = 9e9          # Flag Negative Variance with 9e9
    espec = np.sqrt(espec)
    cd    = cube_header['CD3_3']   # From "per pixel" to "per Angstrom"
    crval =  cube_header['CRVAL3']

    flam_spec = spec * (1. / cd)
    flam_espec = espec * (1. / cd)
    wave = np.zeros(nz)
    for k in range(wave.size):
        wave[k] = crval + cd * k

    return wave, flam_spec, flam_espec


def plot_spectrum(wave, spec, espec, label):
        """
        Plot spectrum
        
        Input:  wavelength, flux, error, string label
        """
        fig = plt.figure()
        plt.title(label)
        plt.plot(wave, spec, 'k', label="flux")
        yset = plt.ylim()
        plt.plot(wave, espec, 'r', alpha = 0.5, label="error") 
        plt.ylim(top = yset[1])
        plt.ylim(bottom = yset[0])
        plt.legend()
        fig.show()
        return

def save_fits_spectrum(wave, spec, espec, header, label):
        """
        Write spectrum and error spectrum to fits files.
        Header is critical for wavelength WCS.
        
        Input:  wavelength, flux, error, FITS header, string label
        """
        # write spectrum and old header
        fits.writeto(label + "_F.fits", spec, header, overwrite=True)
        fits.writeto(label + "_E.fits", espec, header, overwrite=True)

        ######### old way ################################
        #... Use a tuple...                                                                                      
        #crval1 = (hdulist[0].header['CRVAL3'], hdulist[0].header.comments['CRVAL3'])
        #crpix1 = (hdulist[0].header['CRPIX3'], hdulist[0].header.comments['CRPIX3'])
        #cd1_1  = (hdulist[0].header['CD3_3'],  hdulist[0].header.comments['CD3_3'])
        #cname1 = (hdulist[0].header['CNAME3'], hdulist[0].header.comments['CNAME3'])
        #ctype1 = (hdulist[0].header['CTYPE3'], hdulist[0].header.comments['CTYPE3'])
        #cunit1 = (hdulist[0].header['CUNIT3'], hdulist[0].header.comments['CUNIT3'])
        #bunit =  (hdulist[0].header['BUNIT'],  hdulist[0].header.comments['BUNIT'])
        #ttime = (hdulist[0].header['TTIME'], hdulist[0].header.comments['TTIME'])

        #hdu = fits.PrimaryHDU(flam_spec)
        ## write spectrum for inspection   - NEEDS HEADER INFO
        #newhdulist = fits.HDUList([hdu])
        #newhdulist.writeto('test_spec.fits', overwrite=True)
        #newhdulist[0].header['CRVAL1'] = crval1
        #newhdulist[0].header['CRPIX1'] = crpix1
        #newhdulist[0].header['CD1_1'] = cd1_1
        #newhdulist[0].header['CDELT1'] = cd1_1
        #newhdulist[0].header['CNAME1'] = cname1
        #newhdulist[0].header['CTYPE1'] = ctype1
        #newhdulist[0].header['CUNIT1'] = cunit1
        #newhdulist[0].header['BUNIT'] = bunit

        #hdu2 = fits.PrimaryHDU(espec)
        #newhdulist2 = fits.HDUList([hdu2])
        #newhdulist2.writeto('test_espec.fits', overwrite=True)

        return

def save_fits_cube(cube,vcube,cube_header,binmap,binlist,label):
    """
    Save a cube that just shows the spaxels in a region.

    Input:  flux cube, error cube
            cube header
            bin map
            list of bins that define the region
            label for the region
    Ouput:  write FITS files for newcube and error cube
    """
    nz = cube[:,0,0].size
    ny = cube[0,:,0].size
    nx = cube[0,0,:].size
    newcube = np.zeros(nz*ny*nx).reshape(nz,ny,nx)



    for k in range(nz):
        print(str(k)+" out of "+str(nz))
        slice = cube[k,:,:]
        for m in binlist:    
            newslice = np.where(binmap == m, slice, 0)   
            newcube[k,:,:] = newslice + newcube[k,:,:]   # ACTION ITEM;  DO NOT ALLOW OVERLAPPING REGIONS!!!
    fits.writeto('bin_region_' + label + '_cube.fits', newcube, cube_header, overwrite=True)
    return 


def fnd_indx(x,wave):
    """                                                                                                   
    Returns the index of the pixel with wavelength >= wave.                                               
    """
    i = 0
    while(x[i] < wave):
        i = i + 1
    return i

def fit_line(x, y, yerr, w0, wmin, wmax, bmin, bmax):
    """                                                                                                             
    Fit continuum (first order polynomial)                                                                          
    Subtract continuum                                                                                              
    Fit Gaussian line profile                                                                                       
                                                                                                                    
    Fitting functions:                                                                                              
       Define func globally                                                                                          
       func:   single lines                                                                                         
               func = lambda x, a, xcen, sigma:  a * np.exp(-(x-xcen)**2 / (2 * sigma ** 2) )                                  
       func2:  blend

       func_ldl:  blend but with 3 parameters and LDL line ratio

    Return                                                                                                          
       cc:  polynomical coefficients                                                                                
       gc:  gaussian parameters (3 parameters for single line; 4 parameters for blend)                              
       perr: error on gaussian coefficients                                                                         
    """

    ##### Iterate to find boundaries:  [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] 
    #verbose = True                                                                                                 
    verbose = False
    flag_LDL = True # low density limit for O2 doublet.  Leave this ON. Call fit_doublet if NOT LDL.

    while True:   
        # Clip out the section to fit                                                                                   
        ilo = fnd_indx(x,wmin)
        ihi = fnd_indx(x,wmax)
        #if verbose:
        #    print ("ilo = ", ilo, x[ilo])
        #    print ("ihi = ", ihi, x[ihi])
        xsec = np.copy(x[ilo:ihi])  # Does not include x[ihi], which is the first x >= wmax.                            
        ysec = np.copy(y[ilo:ihi])
        esec = np.copy(yerr[ilo:ihi])

        # Clip out the line & fit the background                                                                     
        blo = fnd_indx(x,bmin)
        bhi = fnd_indx(x,bmax)
        #if verbose:
        #    print ("blo = ", blo, x[blo])
        #    print ("bhi = ", bhi, x[bhi])
        xbg = np.append(x[ilo:blo], x[bhi:ihi])
        ybg = np.append(y[ilo:blo], y[bhi:ihi])
        cc = np.polyfit(xbg,ybg,1)  # continuum coefficients (first order polynomial)                                   
        func_bg = np.poly1d(cc)  # function specifying local background                                                 

        # Subtract background                                                                                           
        yfit_cont = func_bg(xsec)  # You can define the background at any desired wavelength                            
        ysub = ysec - yfit_cont

        #####  Make Guess for Line
        amp = np.abs( 10 * ysub.mean() ) # Gaussian amplitude; keep it positive.                                                                   
        gsdev = (bmax - bmin) / 6. # Gaussian stddev (units of x) tied to region excluded from bkgd fit                 

        if (w0 >=3726) and (w0 <= 3730):   # Allow for doublet [OII] 3726.03, 3728.82
            if flag_LDL:
                # 3 parameters                                                                                              
                guess = np.array([amp,w0,gsdev]) # a_amplitude, center, stddev, b_amplitude                         
                plo   = np.array([0, 0, 0])
                phi   = np.array([np.inf, np.inf, np.inf])
                yguess = func_ldl(xsec,guess[0],guess[1],guess[2])
                # use bounds to prevent negative amplitudes
            else:
                # 4 parameters                                                                                              
                guess = np.array([amp,w0,gsdev,amp*0.5]) # a_amplitude, center, stddev, b_amplitude                         
                plo   = np.array([0, 0, 0, 0])
                phi   = np.array([np.inf, np.inf, np.inf, np.inf])
                yguess = func2(xsec,guess[0],guess[1],guess[2],guess[3])
                # use bounds to prevent negative amplitudes
        else:
            # 3 parameters                                                                                              
            #guess = np.array([2e-16,w0,1.]) # amplitude, center, stddev                                                
            guess = np.array([amp,w0,gsdev]) # amplitude, center, stddev                                                
            plo   = np.array([0, 0, 0])
            phi   = np.array([np.inf, np.inf, np.inf])
            yguess = func(xsec,guess[0],guess[1],guess[2])
        if verbose:   # show guess
            yfit = func_bg(xsec)  # You can define the background at any desired wavelength                             
            fig = plt.figure()
            plt.plot(xsec,ysec, 'k:', label="data")
            plt.plot(xsec,yguess, 'g:', label="guess", alpha=0.5)
            plt.plot(xsec, ysub, 'k', label="data-bg")
            yy = [0,0]
            bb = [bmin, bmax]
            ww = [wmin, wmax]
            plt.plot(bb, yy, 'rs')
            plt.plot(ww, yy, 'ks')
            plt.legend()
            fig.show()

        if verbose:
            # Ask user whether a revision is needed. 
            s = input(str(w0) + ":  Fit range [wmin .... bmin XXXXXXX bmax ..... wmax]?  Enter [n] to revise.")
            if s != 'n':
                break
        
            s = input("New wmin [space to skip]: ")
            if s == ' ':
                print("Keep wmin = ", wmin)
            else:
                wmin = float(s)

            s = input("New wmax [space to skip]: ")
            if s == ' ':
                print("Keep wmax = ", wmax)
            else:
                wmax = float(s)

            s = input("New bmin [space to skip]: ")
            if s == ' ':
                print("Keep bmin = ", bmin)
            else:
                bmin = float(s)

            s = input("New bmax [space to skip]: ")
            if s == ' ':
                print("Keep bmax = ", bmax)
            else:
                bmax = float(s)

            continue
        else:
            break  # leave the While Loop if NOT VERBOSE

    if verbose:
        print("Proceeding to fit range [wmin, bmin, bmax, wmax]: ", wmin, bmin, bmax, wmax)

    #####  Fit Line
    if (w0 >3726) and (w0 <= 3730):   # Allow for doublet [OII] 3726.03, 3728.82
        if flag_LDL:
            # 3 parameters                                                                                              
            p, pcov = curve_fit(func_ldl, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)                
        else:
            # 4 parameters                                                                                              
            p, pcov = curve_fit(func2, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)                
    else:
        # 3 parameters                                                                                              
        p, pcov = curve_fit(func, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)                 

    perr = np.sqrt(np.diag(pcov))
    if verbose:
        print('Fit Completed')


    #####  show fit
    if (w0 >=3726) and (w0 <= 3730):   # Allow for doublet [OII] 3726.03, 3728.82
        if flag_LDL:
            # 3 parameters                                                                                              
            yfit_line = func_ldl(xsec,p[0],p[1],p[2])  # blended lines                                                
            xline = np.linspace(bmin,bmax,50)           # high resolution
            yline = func_ldl(xline,p[0],p[1],p[2])  
        else:
            # 4 parameters                                                                                              
            yfit_line = func2(xsec,p[0],p[1],p[2],p[3])  # blended lines                                                
            xline = np.linspace(bmin,bmax,50)           # high resolution
            yline = func2(xline,p[0],p[1],p[2],p[3])  
    else:
        yfit_line = func(xsec,p[0],p[1],p[2])        # single line                                                  
        xline = np.linspace(bmin,bmax,50)           # high resolution
        yline = func(xline,p[0],p[1],p[2])

    if verbose:
        fig = plt.figure()
        #plt.plot(xsec,ysub, 'k', label="data-bg")
        #plt.plot(xsec,yfit_line,'b', label="fit",  alpha=0.5)
        plt.step(xsec,ysub, 'k', label="data-bg")        
        plt.step(xsec,yfit_line,'b', label="fit",  alpha=0.5)
        yzero = np.zeros(yline.size)
        plt.fill_between(xline, yzero, yline, color='blue', where=None, label="fit",  alpha=0.5)
        yy = np.array(plt.ylim())
        plt.plot([p[1], p[1]], yy, "k:")
        #label = "A, xcen, sigma: " + str(p)                                                                        
        label = str(p)
        plt.title(label)
        plt.legend()
        fig.show()
    
    p[2] = np.abs(p[2]) # sigma sign does not matter
    return cc, p, perr

def get_snr_line(wave, flux, err, bmin, bmax):
        """
        Compute S/N ratio across line bandpass.
        """
        i1 = fnd_indx(wave, bmin)
        i2 = fnd_indx(wave, bmax)
        s = 0
        var = 0
        for i in range(i1,i2):
            s = s + flux[i]
            var = var + err[i] ** 2
        err = np.sqrt(var)
        snr = s / err
        return snr


def get_pixel_coords(bin,xiraf,yiraf,dest):
    """
    bin:  2D image array populated by bin number
    (xiraf, yiraf):  center of object in SAOimage ds9 display
    
    Returns two 2D image arrays:  
        radius relative to (xiraf, yiraf) and 
        angle theta (relative to 3pm)
    """
    # Populate the radius map
    x0 = xiraf - 1
    y0 = yiraf - 1
    nx = bin[0,:].size
    ny = bin[:,0].size
    ximage = np.zeros(nx * ny).reshape(ny,nx)
    yimage = np.zeros(nx * ny).reshape(ny,nx)

    # assign pixels coordinates and map radius [pixel units]
    for i in range(nx):
        yimage[:,i] = np.arange(ny, dtype = int)
    for j in range(ny):
        ximage[j,:] = np.arange(nx, dtype = int)
    radius = np.sqrt( (ximage - x0)**2 + (yimage - y0) **2 )
    fits.writeto(dest+'radius.fits', radius, image_header, overwrite=True)
    fits.writeto(dest+'ximage.fits', ximage, image_header, overwrite=True)
    fits.writeto(dest+'yimage.fits', yimage, image_header, overwrite=True)

    # map angle (-180, +180) from x-axis
    # See clm_arctan2.py example in ~/Software/python_scripts/clm/
    theta = np.zeros(nx * ny).reshape(ny,nx)
    for i in range(nx):
        for j in range(ny):
            theta[j,i] = np.arctan2( (yimage[j,i] - y0), (ximage[j,i] - x0) ) * 180 / np.pi    ## [degrees]
    fits.writeto(dest+'theta.fits', theta, image_header, overwrite=True)

    return radius, theta


def get_fit5007(wk0, flux, err, wflag):
    """
    Wavelength is rest frame.
    Flux is observed frame - needs to be fixed OR fit in observed frame.
    """
    if wflag == 'vac':
        wline = 5008.24 
    else:
        wline= 5006.84 # use air

    ####### [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] ######## 
    wmin = 4970
    bmin = 5000
    bmax = 5012
    wmax = 5030
    try:
        c5007, p5007, e5007 = fit_line(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
    except RuntimeError:
        print("Bin", m, "no 5007 fit.")
        c5007 = np.array([0,0])
        p5007 = np.array([0,5006.84,1])
        e5007 = 10 * p5007
    myline = func(wk0, *p5007) # fitted line
    p1 = np.poly1d(c5007)
    myfit  = func(wk0, *p5007) + p1(wk0)# fitted line + continuum
    flux5007 = np.sqrt(2 * np.pi) * p5007[0] * (1 + zspec) * p5007[2]  # Area = sqrt(2 pi) * amplitude * stddev
    snr5007 = get_snr_line(wk0,flux,err,bmin,bmax)

    return flux5007, snr5007

def get_line5007(wk0, flux, err, wflag):
    """
    Wavelength is rest frame.
    Flux is observed frame - needs to be fixed OR fit in observed frame.
    """
    if wflag == 'vac':
        wline = 5008.24 
    else:
        wline= 5006.84 # use air

    ####### [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] ######## 
    wmin = 4970
    bmin = 5000
    bmax = 5012
    wmax = 5030
    try:
        c5007, p5007, e5007 = fit_line(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
    except RuntimeError:
        print("Bin", m, "no 5007 fit.")
        c5007 = np.array([0,0])
        p5007 = np.array([0,5006.84,1])
        e5007 = 10 * p5007
    myline = func(wk0, *p5007) # fitted line
    p1 = np.poly1d(c5007)
    myfit  = func(wk0, *p5007) + p1(wk0)# fitted line + continuum
    flux5007 = np.sqrt(2 * np.pi) * p5007[0] * (1 + zspec) * p5007[2]  # Area = sqrt(2 pi) * amplitude * stddev
    snr5007 = get_snr_line(wk0,flux,err,bmin,bmax)

    return flux5007, snr5007, p5007,e5007,wline


def get_fit4959(wk0, flux, err, wflag):
    """
    Wavelength is rest frame.
    Flux is observed frame - needs to be fixed OR fit in observed frame.
    """
    if wflag == 'vac':
        wline = 4960.30 
    else:
        wline= 4958.92 # use air

    ####### [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] ######## 
    wmin = 4920
    bmin = 4954
    bmax = 4961
    wmax = 5000
    try:
        c4959, p4959, e4959 = fit_line(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
    except RuntimeError:
        print("Bin", m, "no 4959 fit.")
        c4959 = np.array([0,0])
        p4959 = np.array([0,4958.92,1])
        e4959 = 10 * p4959
    myline = func(wk0, *p4959) # fitted line
    p1 = np.poly1d(c4959)
    myfit  = func(wk0, *p4959) + p1(wk0)# fitted line + continuum
    flux4959 = np.sqrt(2 * np.pi) * p4959[0] * (1 + zspec) * p4959[2]  # Area = sqrt(2 pi) * amplitude * stddev
    snr4959 = get_snr_line(wk0,flux,err,bmin,bmax)

    return flux4959, snr4959


def get_fit3727(wk0, flux, err, wflag, flag_LDL):
    """
    Wavelength is rest frame for the higher J line.
    Flux is observed frame - needs to be fixed OR fit in observed frame.

    Return:  line flux, line snr, doublet ratio, s/n ratio on doublet ratio
    """
    if wflag == 'vac':
        wline = 3729.88
    else:
        wline= 3728.82 # use air

    ####### [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] ######## 
    wmin = 3690
    bmin = 3720
    bmax = 3732
    wmax = 3750
    flag3=False
    dratio=0
    snratio=0
    try:
        if flag_LDL:   # (3 parameters)
            c3727, p3727, e3727 = fit_line(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
        else:   # fit the doublet (4 parameters)
            c3727, p3727, e3727, flag3, dratio, snratio = fit_doublet(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
    except RuntimeError:
        print("Bin", m, "no 3727 fit.")
        c3727 = np.array([0,0])
        if flag_LDL:
           p3727 = np.array([0,wline,1])   # 3 parameters
        else:
           p3727 = np.array([0,wline,1,0])   # 4 parameters
        e3727 = 10 * p3727

    p1 = np.poly1d(c3727)
    if flag3 == 'LDL':
        myline = func_ldl(wk0, *p3727) # fitted line
        flux3727 = np.sqrt(2 * np.pi) * p3727[0] * (1 + zspec) * p3727[2] + np.sqrt(2 * np.pi) * p3727[0] / 1.47 * (1 + zspec) * p3727[2]     # Area for doublet
        snr3727 = get_snr_line(wk0,flux,err,bmin,bmax)
    elif flag3 == 'HDL':
        myline = func_hdl(wk0, *p3727) # fitted line
        flux3727 = np.sqrt(2 * np.pi) * p3727[0] * (1 + zspec) * p3727[2] + np.sqrt(2 * np.pi) * p3727[0]  / 0.29 * (1 + zspec) * p3727[2]     # Area for doublet
        snr3727 = get_snr_line(wk0,flux,err,bmin,bmax)
    else:  # 4 parameters
        myline = func2(wk0, *p3727) # fitted line
        flux3727 = np.sqrt(2 * np.pi) * p3727[0] * (1 + zspec) * p3727[2] + np.sqrt(2 * np.pi) * p3727[3] * (1 + zspec) * p3727[2]     # Area for doublet
        snr3727 = get_snr_line(wk0,flux,err,bmin,bmax)
    myfit  = myline + p1(wk0)# fitted line + continuum


    return flux3727, snr3727, dratio, snratio

def fit_doublet(x, y, yerr, w0, wmin, wmax, bmin, bmax):
    """                                                                                                             
    Fit continuum (first order polynomial)                                                                          
    Subtract continuum                                                                                              
    Fit doublet with two Gaussian line profiles                                                                                                                    
    Fitting functions:                                                                                              
       Define func globally                                                                                          

       func3 (5 parameters): Ar4 + HeI blend with independent amplitudes

       func2 (4 parameters): O2 blend with independent amplitudes
       func_ldl (3 parameters):  O2 blend but with 3 parameters and LDL line ratio
       func_hdl (3 parameters):  O2 blend but with 3 parameters and LDL line ratio

    Return                                                                                                          
       cc:  polynomical coefficients                                                                                
       gc:  gaussian parameters (3 parameters for single line; 4 parameters for blend)                              
       perr: error on gaussian coefficients                                                                         
       flag3: identify 3 parameter doublet fits as 'LDL' or 'HDL'
    """

    ##### Iterate to find boundaries:  [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] 
    #verbose = True                                                                                                 
    verbose = False

    res = 3600 # Spectral resolution (for initial stddev guess)

    while True:   
        # Clip out the section to fit                                                                                   
        ilo = fnd_indx(x,wmin)
        ihi = fnd_indx(x,wmax)
        #if verbose:
        #    print ("ilo = ", ilo, x[ilo])
        #    print ("ihi = ", ihi, x[ihi])
        xsec = np.copy(x[ilo:ihi])  # Does not include x[ihi], which is the first x >= wmax.                            
        ysec = np.copy(y[ilo:ihi])
        esec = np.copy(yerr[ilo:ihi])

        # Clip out the line & fit the background                                                                     
        blo = fnd_indx(x,bmin)
        bhi = fnd_indx(x,bmax)
        #if verbose:
        #    print ("blo = ", blo, x[blo])
        #    print ("bhi = ", bhi, x[bhi])
        xbg = np.append(x[ilo:blo], x[bhi:ihi])
        ybg = np.append(y[ilo:blo], y[bhi:ihi])
        cc = np.polyfit(xbg,ybg,1)  # continuum coefficients (first order polynomial)                                   
        func_bg = np.poly1d(cc)  # function specifying local background                                                 

        # Subtract background                                                                                           
        yfit_cont = func_bg(xsec)  # You can define the background at any desired wavelength                            
        ysub = ysec - yfit_cont

        #####  Make Guess for Line
        amp = np.abs( 10 * ysub.mean() ) # Gaussian amplitude; keep it positive.
        gsdev = w0 / res / 2.35 # fwhm / 2.35 = Gaussian stddev (units of x) tied to region excluded from bkgd fit

        if (w0 >=3726) and (w0 <= 3730):   # Allow for doublet [OII] 3726.03, 3728.82
            # 4 parameters                                                                                              
            guess = np.array([amp,w0,gsdev,amp*2/3]) # a_amplitude, center, stddev, b_amplitude                         
            plo   = np.array([0, 0, 0, 0]) # use bounds to prevent negative amplitudes
            phi   = np.array([np.inf, np.inf, np.inf, np.inf])
            yguess = func2(xsec,guess[0],guess[1],guess[2],guess[3])
        if (w0 >=4710) and (w0 <= 4713):   # Allow for doublet [ArIV] 4711, 4740 + HeI 4713 blend
            # 5 parameters                                                                                              
            guess = np.array([amp,w0,gsdev,amp*2/3,amp*0.5]) # a_amplitude, center, stddev, b_amplitude, c_amplitude                         
            plo   = np.array([0, 0, 0, 0, 0]) # use bounds to prevent negative amplitudes
            phi   = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
            yguess = func3(xsec,guess[0],guess[1],guess[2],guess[3],guess[4]) #Evaluate guess
            
        #########  make a function for the revision ###############################################
        if verbose:   # show guess
            yfit = func_bg(xsec)  # You can define the background at any desired wavelength                             
            fig = plt.figure()
            plt.plot(xsec,ysec, 'k:', label="data")
            plt.plot(xsec,yguess, 'g:', label="guess", alpha=0.5)
            plt.plot(xsec, ysub, 'k', label="data-bg")
            yy = [0,0]
            bb = [bmin, bmax]
            ww = [wmin, wmax]
            plt.plot(bb, yy, 'rs')
            plt.plot(ww, yy, 'ks')
            plt.legend()
            fig.show()

        if verbose:
            # Ask user whether a revision is needed. 
            s = input(str(w0) + ":  Fit range [wmin .... bmin XXXXXXX bmax ..... wmax]?  Enter [n] to revise.")
            if s != 'n':
                break
        
            s = input("New wmin [space to skip]: ")
            if s == ' ':
                print("Keep wmin = ", wmin)
            else:
                wmin = float(s)

            s = input("New wmax [space to skip]: ")
            if s == ' ':
                print("Keep wmax = ", wmax)
            else:
                wmax = float(s)

            s = input("New bmin [space to skip]: ")
            if s == ' ':
                print("Keep bmin = ", bmin)
            else:
                bmin = float(s)

            s = input("New bmax [space to skip]: ")
            if s == ' ':
                print("Keep bmax = ", bmax)
            else:
                bmax = float(s)

            continue
        else:
            break  # leave the While Loop if NOT VERBOSE

    if verbose:
        print("Proceeding to fit range [wmin, bmin, bmax, wmax]: ", wmin, bmin, bmax, wmax)
    ##################### end of bandpass revision function


    #####  Fit Line & Find Doublet Ratio
    xline = np.linspace(bmin,bmax,500)           # high resolution for smooth fitted profile

    if (w0 >=3726) and (w0 <= 3730):   # Allow for doublet [OII] 3726.03, 3728.82
        ratio_ldl = 1.47  # limits for PyNeb
        ratio_hdl = 0.29
        # 4 parameters                                                                                              
        p, pcov = curve_fit(func2, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
        yfit_line = func2(xsec,p[0],p[1],p[2],p[3])  # blended lines                                                
        yline = func2(xline,p[0],p[1],p[2],p[3])  
        # revise fit if doublet ratio is outside PyNeb's bounds
        flag3 = False
        test_ratio = p[0] / p[3]
        if (test_ratio < ratio_hdl) or (test_ratio > ratio_ldl):   # use limits on physical ratio (per PyNeb)
            guess = np.array([amp,w0,gsdev]) # amplitude, center, stddev (3 parameter fit)
            plo   = np.array([0, 0, 0])
            phi   = np.array([np.inf, np.inf, np.inf])
            yguess = func(xsec,guess[0],guess[1],guess[2])
            if test_ratio > ratio_ldl:   # use LDL
                flag3 = 'LDL' # 3 parameter fit
                p, pcov = curve_fit(func_ldl, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
                yfit_line = func_ldl(xsec,p[0],p[1],p[2])  # blended lines                                                
                yline = func_ldl(xline,p[0],p[1],p[2])  
                dratio = ratio_ldl
            elif (test_ratio < ratio_hdl):  # use HDL
                flag3 = 'HDL' # 3 parameter fit
                p, pcov = curve_fit(func_hdl, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
                yfit_line = func_hdl(xsec,p[0],p[1],p[2])  # blended lines                                                
                yline = func_hdl(xline,p[0],p[1],p[2])  
                dratio = ratio_hdl
            perr = np.sqrt(np.diag(pcov))
            eratio = perr[0] / p[0] * dratio    # Use the percentage error on the A-amplitude for dratio error
        else:  # 4 parameter fit
            dratio = p[0] / p[3]  # F(3729)/F(3726)
            perr = np.sqrt(np.diag(pcov))
            eratio = np.sqrt( (perr[3] / p[3])**2 + (perr[0] / p[0])**2 ) * dratio

    if (w0 >=4710) and (w0 <= 4713):   # Allow for doublet [ArIV] 4711, 4740 + He I blend
        ratio_ldl = 1.37  # limits for PyNeb
        ratio_hdl = 0.76
        # 4 parameters                                                                                              
        p, pcov = curve_fit(func3, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
        yfit_line = func3(xsec,p[0],p[1],p[2],p[3],p[4])  # blended lines                                                
        yline = func3(xline,p[0],p[1],p[2],p[3],p[4])  
        # revise fit if doublet ratio is outside PyNeb's bounds
        flag3 = False
        test_ratio = p[0] / p[3]
        if (test_ratio < ratio_hdl) or (test_ratio > ratio_ldl):   # use limits on physical ratio (per PyNeb)
            guess = np.array([amp,w0,gsdev,0.5*amp]) # amplitude, center, stddev (4 parameter fit)
            plo   = np.array([0, 0, 0, 0])
            phi   = np.array([np.inf, np.inf, np.inf, np.inf])
            if test_ratio > ratio_ldl:   # use LDL
                flag3 = 'LDL' # 4 parameter fit
                yguess = func3_ldl(xsec,guess[0],guess[1],guess[2],guess[3])
                p, pcov = curve_fit(func3_ldl, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
                yfit_line = func3_ldl(xsec,p[0],p[1],p[2],p[3])  # Fixed Ar4 ratio plus free HeI
                yline = func3_ldl(xline,p[0],p[1],p[2],p[3])  
                dratio = ratio_ldl
            elif (test_ratio < ratio_hdl):  # use HDL
                flag3 = 'HDL' # 4 parameter fit
                yguess = func3_hdl(xsec,guess[0],guess[1],guess[2],guess[3])
                p, pcov = curve_fit(func3_hdl, xsec, ysub, guess, esec, bounds = [plo,phi])  # y is continuum subtracted (important)
                yfit_line = func3_hdl(xsec,p[0],p[1],p[2],p[3])  # Fixed Ar4 ratio plus free HeI
                yline = func3_hdl(xline,p[0],p[1],p[2],p[3])  
                dratio = ratio_hdl
            perr = np.sqrt(np.diag(pcov))
            eratio = perr[0] / p[0] * dratio    # Use the percentage error on the A-amplitude for dratio error
        else:  # 5 parameter fit
            dratio = p[0] / p[3]  # F(4711)/F(4740)
            perr = np.sqrt(np.diag(pcov))
            eratio = np.sqrt( (perr[3] / p[3])**2 + (perr[0] / p[0])**2 ) * dratio

    snratio = dratio / eratio

    if verbose:
        print('Fit Completed')


    #if True:
    if verbose:
        fig = plt.figure()
        #plt.plot(xsec,ysub, 'k', label="data-bg")
        #plt.plot(xsec,yfit_line,'b', label="fit",  alpha=0.5)
        plt.step(xsec,ysub, 'k', label="data-bg")        
        plt.step(xsec,yfit_line,'b', label="fit",  alpha=0.5)
        yzero = np.zeros(yline.size)
        plt.fill_between(xline, yzero, yline, color='blue', label="fit",  where=None, alpha=0.5)
        #label = "A, xcen, sigma: " + str(p)                                                                        
        label = str(p)
        plt.title(label)
        plt.legend()
        fig.show()
    
    p[2] = np.abs(p[2]) # sigma sign does not matter
    return cc, p, perr, flag3, dratio, snratio

def get_fitAr4(wk0, flux, err, wflag, flag_LDL):
    """
    Wavelength is rest frame for the higher J line.
    Flux is observed frame - needs to be fixed OR fit in observed frame.

    Return:  line flux, line snr, doublet ratio, s/n ratio on doublet ratio
    """
    if wflag == 'vac':
        wline = 4712.67
    else:
        wline= 4711.35 # use air

    ####### [wmin] .. continuum ...  [bmin]  ... line ... [bmax] ... continuum .... [wmax] ######## 
    wmin = 4700
    bmin = 4707
    bmax = 4744
    wmax = 4760
    flag3=False
    try:
        if flag_LDL:   # (3 parameters)
            print ('flag_LDL not supported for Ar4')
            exit()
        else:   # fit the doublet + He I (5 parameters)
            cAr4, pAr4, eAr4, flag3, dratio, snratio = fit_doublet(wk0, flux, err, wline, wmin, wmax, bmin, bmax)
    except RuntimeError:
        print("Bin", m, "no Ar4 fit.")
        cAr4 = np.array([0,0])
        if flag_LDL:
            print ('flag_LDL not supported for Ar4')
            exit()
        else:  # 5 parameters
           pAr4 = np.array([wline,0,1,0,0,0])   
        eAr4 = 10 * pAr4

    p1 = np.poly1d(cAr4)
    if flag3 == 'LDL':
        myline = func3_ldl(wk0, *pAr4) # fitted line
        # Area for doublet; note that you don't want the He I area, pAr4[4]
        fluxAr4 = np.sqrt(2 * np.pi) * pAr4[0] * (1 + zspec) * pAr4[2] + np.sqrt(2 * np.pi) * pAr4[0] / 1.37  * (1 + zspec) * pAr4[2]
        snrAr4 = get_snr_line(wk0,flux,err,bmin,bmax)
    elif flag3 == 'HDL':
        myline = func3_hdl(wk0, *pAr4) # fitted line
        # Area for doublet; note that you don't want the He I area, pAr4[4]
        fluxAr4 = np.sqrt(2 * np.pi) * pAr4[0] * (1 + zspec) * pAr4[2] + np.sqrt(2 * np.pi) * pAr4[0] / 0.76  * (1 + zspec) * pAr4[2]
        snrAr4 = get_snr_line(wk0,flux,err,bmin,bmax)
    else:  # 5 parameters
        myline = func3(wk0, *pAr4) # fitted line
        # Area for doublet; note that you don't want the He I area, pAr4[4]
        fluxAr4 = np.sqrt(2 * np.pi) * pAr4[0] * (1 + zspec) * pAr4[2] + np.sqrt(2 * np.pi) * pAr4[3] * (1 + zspec) * pAr4[2]
        snrAr4 = get_snr_line(wk0,flux,err,bmin,bmax)
    myfit  = myline + p1(wk0)# fitted line + continuum


    return fluxAr4, snrAr4, dratio, snratio


###### Main Program ################################
# Sum spectra for all pixels in a bin.
##################################################################################
##bin_file = 'j1044+0353/assigned3727_j1044.fits'                    # Assigned Bins, as ID[x,y] Used O2 3727
#bin_file = 'j1044+0353/assigned4740_j1044.fits'                    # Assigned Bins, as ID[x,y] Used Ar4 4740 map - snr5
#cube_file = 'j1044+0353/j1044+0353_addALL_icubes.fits'   # Flux Cube
#vcube_file = 'j1044+0353/j1044+0353_addALL_vcubes.fits'  # Variance Cube
#zspec = 0.01300   #KCWI redshift not SDSS 0.01287 
#xiraf = 62.98
#yiraf = 59.96
#
##bin_file = 'j0248-0817/assigned3727_j0248.fits'                 # Assigned Bins, as ID[x,y] Used O2 3727
#bin_file = 'j0248-0817/assigned4741_j0248.fits'                 # Assigned Bins, as ID[x,y] Used O2 3727cube_file = 'j0248-0817/J024815-081723_icubes.wc.c.fits'  # Flux Cube
#cube_file = 'j0248-0817/J024815-081723_icubes.wc.c.fits' # Variance Cube
#vcube_file = 'j0248-0817/J024815-081723_vcubes.wc.c.fits' # Variance Cube
#zspec = 0.004722
#xiraf = 79.8
#yiraf = 70.5
#
'''
bin_file = '/Volumes/TOSHIBA EXT/MARTINLAB/target5/z_image.j1238+1009_main_icubes3727_assigned.fits'    # Assigned Bins, as ID[x,y] Used O2 3727
#bin_file = 'j0823+0313/assigned4740_j0823.fits'    # Assigned Bins, as ID[x,y] ArIV
cube_file = '/Volumes/TOSHIBA EXT/MARTINLAB/original/j1238+1009_main_icubes.fits' # Flux Cube
vcube_file = '/Volumes/TOSHIBA EXT/MARTINLAB/original/j1238+1009_main_vcubes.fits' # Variance Cube
dest="/Volumes/TOSHIBA EXT/MARTINLAB/j1238/"
'''
bin_file = '/Volumes/TOSHIBA EXT/MARTINLAB/target5/z_image.j0823+0313_17frames_icubes3727_assigned.fits'    # Assigned Bins, as ID[x,y] Used O2 3727
#bin_file = 'j0823+0313/assigned4740_j0823.fits'    # Assigned Bins, as ID[x,y] ArIV
cube_file = '/Volumes/TOSHIBA EXT/MARTINLAB/original/j0823+0313_17frames_icubes.fits' # Flux Cube
vcube_file = '/Volumes/TOSHIBA EXT/MARTINLAB/original/j0823+0313_17frames_vcubes.fits' # Variance Cube
dest="/Volumes/TOSHIBA EXT/MARTINLAB/j0823/"

zspec = 0.009864
xiraf = 119
yiraf = 96
#
#bin_file = 'j1238+1009/assigned3727_j1238.fits'
##bin_file   = 'j1238+1009/assigned4740_j1238.fits'
#cube_file  = 'j1238+1009/j1238+1009_main_icubes.fits'
#vcube_file = 'j1238+1009/j1238+1009_main_vcubes.fits'
#zspec = 0.003928
#xiraf = 89.5
#yiraf = 66.5

print("line 881")
##################################################################################
wflag = 'air'
#verbose = True
verbose = False
#flag_LDL = True # low density limit for O2 doublet
flag_LDL = False
flag_Ar4 = False  # using Ar4 bin assignments?  Otherwise assume O2 bin assignments

hdulist = fits.open(bin_file)       
bin = hdulist[0].data               # [y_image, x_image]

hdulist = fits.open(vcube_file)    
vcube = hdulist[0].data

hdulist = fits.open(cube_file)     
cube_header = fits.getheader(cube_file)
cube = hdulist[0].data

# Set the WCS for the wavelength direction in 1D
spec_header = cube_header.copy()     # does this need to be deep copy?                  
spec_header['CRVAL1'] = cube_header['CRVAL3']   
spec_header['CD1_1'] = cube_header['CD3_3']     
# Set the WCS for an image direction in 2D
image_header = cube_header.copy()      # does this need to be deep copy?                  
image_header['PHOTFLAM'] = 1
image_header['WCSDIM'] = 2
image_header.remove('CTYPE3')
image_header.remove('CUNIT3')
image_header.remove('CNAME3')
image_header.remove('CRVAL3')
image_header.remove('CRPIX3')
image_header.remove('CD3_3')

print("line 915")

#show_bins(bin)  # plot the map of all bins
goodbins = np.where(bin > 0, bin, 0)   
show_bins(goodbins) # plot the map of all the good bins (those with a significant signal)

print("line 921")

# select bins for spectral extraction
id, nid = np.unique(goodbins, return_counts=True)   # Find unique bin numbers and count their spaxels
if verbose:
    fig = plt.figure()
    plt.plot(id,nid, 'sk')
    fig.show()

print("line 933")

binlist = list(id[1:])  # list of bins for spectral extraction; breaks for m=0, so skip first entry

# Example O2 bins
#binlist = list(id[599:])  # list of bins for spectral extraction; breaks for m=0, so skip first entry
#binlist = [1573]  #J1044
#binlist = [272,1964,548,2018,1573]  #J1044
#binlist = [1610,761]  #J1044
#binlist = [861, 2197, 1167, 2166, 446]
#binlist = [4365]  # O2 bin for J1238
    
# Example Ar4 bins
#binlist = [93,30,150]  # Ar4 bin for J1044


nspec = len(binlist)
nz = cube[:,0,0].size
ny = cube[0,:,0].size
nx = cube[0,0,:].size
'''
allwave = np.zeros(nspec*nz).reshape((nspec,nz))
allflux = np.zeros(nspec*nz).reshape((nspec,nz))
allerr = np.zeros(nspec*nz).reshape((nspec,nz))

'''
allmask = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
#save_fits_cube(cube, vcube, cube_header, bin, binlist, 'test')

print("line 959")

'''
o32 = []
o32snr = []

listdr = []
listdrsnr = []

if flag_Ar4:
    listAr4 = []
    listAr4snr = []

listall = []
listallsnr = []
'''

shift=[]
shifte=[]
sigma=[]
sigmae=[]

nm = 0  # index counter for binlist
for m in binlist:
    print ("Extracting bin ", m,"out of",len(binlist))
    mask = np.where(bin == m, 1, 0)   # ZERO is reserved for pixels where binning did not reach the threshold    
    wave, flam_spec, flam_espec = spec_extract(cube, vcube, mask)

    '''
    if verbose:
        fits.writeto('bin' + str(m) + '_mask.fits', mask, image_header, overwrite=True)  # output to visualize bin
        save_fits_spectrum(wave, flam_spec, flam_espec, spec_header, label="spec1d_" + str(m))
    if verbose:
        plot_spectrum(wave, flam_spec, flam_espec, label= "bin" + str(m))
    
    allwave[nm,:]   = np.copy(wave)
    allflux[nm,:]   = np.copy(flam_spec)
    allerr[nm,:]    = np.copy(flam_espec)
    '''
    allmask[nm,:,:] = np.copy(mask)
    wk0 = wave / (1 + zspec)  # rest-frame wavelength
    print ("Fitting bin ", m,"out of",len(binlist))
    flux5007, snr5007,p5007,e5007,wline = get_line5007(wk0, flam_spec, flam_espec, wflag)
    redshift=(p5007[1]-wline)/wline
    shift.append(redshift)
    shifte.append(e5007[1]/wline)
    sigma.append(p5007[2])
    sigmae.append(e5007[2])

    nm = nm + 1



# FIT THE EMISSION LINES HERE
# compute O32
# o32.append(newvalue)
#line = ["O3_4959", "O3_5007", "O2"] 
#w0 = [4958.92, 5006.84, 3728.82] # rest-frame wavelengths  3726.03, 3728.82
3
wavea,flam_speca,flam_especa=spec_extract(cube, vcube, np.where(bin==bin,bin,0))
flux5007a, snr5007a,p5007a,e5007a,wlinea = get_line5007(wavea / (1 + zspec), flam_speca, flam_especa, wflag)
redshifta=(p5007a[1]-wlinea)/wlinea
print(flux5007a)
print(snr5007a)
print(redshifta)
shiftdif=[a-redshifta for a in shift]

print("line 1036")

mastershift = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    mastershift[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * shift[nm]
mapshift = np.sum(mastershift, axis=0)
fits.writeto(dest+'shift_bin.fits', mapshift, image_header, overwrite=True) 
mastershifte = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    mastershifte[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * shifte[nm]
mapshifte = np.sum(mastershifte, axis=0)
fits.writeto(dest+'shifte_bin.fits', mapshifte, image_header, overwrite=True)
mastersigma= np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    mastersigma[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * sigma[nm]
mapsigma = np.sum(mastersigma, axis=0)
fits.writeto(dest+'sigma_bin.fits', mapsigma, image_header, overwrite=True) 
mastersigmae= np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    mastersigmae[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * sigmae[nm]
mapsigmae = np.sum(mastersigmae, axis=0)
fits.writeto(dest+'sigmae_bin.fits', mapsigmae, image_header, overwrite=True) 

mastershiftdif = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    mastershiftdif[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * shiftdif[nm]
mapshiftdif = np.sum(mastershiftdif, axis=0)
fits.writeto(dest+'shiftdif_bin.fits', mapshiftdif, image_header, overwrite=True)  

'''
# Populate the O32 map
masterO32 = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    masterO32[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * o32[nm]
mapO32 = np.sum(masterO32, axis=0)
fits.writeto(dest+'O32_bin.fits', mapO32, image_header, overwrite=True)  


masterO32snr = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):   # each entry in binlist
    masterO32snr[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * o32snr[nm]
mapO32snr = np.sum(masterO32snr, axis=0)
fits.writeto(dest+'O32snr_bin.fits', mapO32snr, image_header, overwrite=True)  


masterO2DR = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):
    masterO2DR[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * listdr[nm]
mapO2DR = np.sum(masterO2DR, axis=0)
fits.writeto(dest+'O2DR_bin.fits', mapO2DR, image_header, overwrite=True)  

masterO2DRsnr = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
for nm in range(nspec):
    masterO2DRsnr[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * listdrsnr[nm]
mapO2DRsnr = np.sum(masterO2DRsnr, axis=0)
fits.writeto(dest+'O2DRsnr_bin.fits', mapO2DRsnr, image_header, overwrite=True)  


if flag_Ar4:
    masterAr4DR = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
    for nm in range(nspec):
        masterAr4DR[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * listAr4[nm]
    mapAr4DR = np.sum(masterAr4DR, axis=0)
    fits.writeto(dest+'Ar4DR_bin.fits', mapAr4DR, image_header, overwrite=True)  

    masterAr4DRsnr = np.zeros(nspec*ny*nx).reshape((nspec,ny,nx))
    for nm in range(nspec):
        masterAr4DRsnr[nm,:,:] = np.where(allmask[nm,:,:] > 0, 1, 0) * listAr4snr[nm]
    mapAr4DRsnr = np.sum(masterAr4DRsnr, axis=0)
    fits.writeto(dest+'Ar4DRsnr_bin.fits', mapAr4DRsnr, image_header, overwrite=True)  
'''


print("line 1081")


# Map the radius pixel by pixel
radius, theta = get_pixel_coords(bin,xiraf,yiraf,dest)

# compute radius of each bin 
bin_radii = []
bin_radii_err = []
bin_theta = []
for m in binlist:
    mask = np.where(bin == m, bin, 0)  # mask for the bin
    coords = [(j,i) for j in range(mask[:,0].size) for i in range(mask[0,:].size) if mask[j,i] !=0]  # select spaxels
    rlist = [radius[j,i] for (j,i) in coords]
    thlist = [theta[j,i] for (j,i) in coords]
    # Weighted average requires subroutine. You do not have line flux at each pixel, only for each bin.
    # You could use inverse variance at each pixel at line center (or over some fixed line width) from vcube; but
    # you still need to identify the index of the line center in the cube.
    bin_radii.append(np.mean(rlist))
    bin_theta.append(np.mean(thlist))
    if len(rlist) <= 1:
        bin_radii_err.append(0.5)   # assign half a pixel
    else:
        bin_radii_err.append(np.std(rlist))  



# Save Output
data = Table([bin_radii, bin_radii_err, bin_theta, shift, shifte,shiftdif,sigma,sigmae, binlist], names=['R(pix)', 'errR(pix)', "theta", 'shift', 'shifterr','shiftdif','sigma','sigmaerr', 'bin'])
ascii.write(data, dest+"output_bin_R_shift.txt", overwrite = True)

'''
if flag_Ar4:
    data = Table([bin_radii, bin_radii_err, bin_theta, listall, listallsnr, listAr4, listAr4snr, binlist], names=['R(pix)', 'errR(pix)', "theta", 'O2DR', 'snrO2DR', 'Ar4DR', 'snrAr4DR', 'bin'])
    ascii.write(data, dest+"output_bin_R_den.txt", overwrite = True)
'''
print("line 1117")


#################################################################################

# old way
#hdu = fits.PrimaryHDU(newcube)
#newhdulist = fits.HDUList([hdu])
#newhdulist.writeto('test_cube.fits', overwrite=True)


#hdu = fits.PrimaryHDU(mask)
#newhdulist = fits.HDUList([hdu])
#newhdulist.writeto('bin' + str(m) + '_mask.fits', overwrite=True)

print("total processing time",time.time()-start)

