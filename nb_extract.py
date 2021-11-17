# clm: 12/10/2020
#       - Add PHOTFLAM keyword
# clm: 07/16/2020
#       - Change output file names to have the wavelength of 'wrest' instead of defaulting to 'wvac'
#       - Apply to J104457+035313
#       - Change output names to use wrest instead of wvac
#       - Moved upper background window; separated it from NB bandpass. See:  k2a_prime, w2a_prime.
# clm:  02/07/2020
#       - Adjust range() limits for background estimate.
#       - Adjust wavelength indexing; now compatible wtih CWITools CRVAL3 = 0
# clm:  9/02/2019
#       - Apply to the *wc.fits cubes produced by CWITools.
# clm:  8/12/2019
#       - Added plot of background estimate
#       - NOTE: error image contains the error from the source image and the continuum subtraction.
#       - BG 'background' is really the 'continuum estimate' as the sky background is included in '_vcubes.fits'.
# clm:  3/26/2019
#       - Extract a narrowband image
# clm:  1/11/2017
#       - Extract an image from a KCWI datacube
#
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


########################################################
# Choose KCWI reduced data cube
# Choose extraction location (LL corner in IRAF/DS9)
# Set filename for extracted (summed) spectrum
########################################################

###########################################################################
###########################################################################
# These are flags for excluding part of the BG bandpass
flag_hi = False
flag_lo = False

inputlist=[]
errlist=[]
zsplist=[]
xclist=[]
yclist=[]
source="/Volumes/TOSHIBA EXT/MARTINLAB/original/"
dest="/Volumes/TOSHIBA EXT/MARTINLAB/unbinned/"

# Nite 2 (0.5 A/pix) j1044+0353
input_cube = "j1044+0353_addALL_icubes.fits"
err_cube   = "j1044+0353_addALL_vcubes.fits"
zspec = 0.013000
xc = 63  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 60

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

## Nite 2 (0.5 A/pix) j0248+0817
input_cube = "J024815-081723_icubes.wc.c.fits"
err_cube   = "J024815-081723_vcubes.wc.c.fits"
zspec = 0.004562
xc = 63  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 60

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

## Nite 2 (0.5 A/pix) j0944-0039
input_cube = "j0944-0039_addALL_1200_icubes.fits"
err_cube   = "j0944-0039_addALL_1200_vcubes.fits"
zspec = 0.004776
xc = 63  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 60

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

# Nite 2 (0.5 A/pix) j1418+2101
input_cube = "j1418+2101_add00055_icubes.fits"
err_cube   = "j1418+2101_add00055_vcubes.fits"
zspec = 0.008547
xc = 36  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 73

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

input_cube = "j1418+2101_add1200_icubes.fits"
err_cube   = "j1418+2101_add1200_vcubes.fits"
zspec = 0.008547
xc = 50  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 73

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

##Nite 2 (0.5 A/pix) j0837+5138
input_cube = "j0837+5138_all1200_icubes.fits"
err_cube   = "j0837+5138_all1200_vcubes.fits"
zspec = 0.002451
xc = 33 # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 71

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

#Nite 2 (0.5 A/pix) j1016+3754
input_cube = "j1016+3754_addALL1200_icubes.fits"
err_cube   = "j1016+3754_addALL1200_vcubes.fits"
zspec = 0.003879
xc = 76 # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 73

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

## Nite 2 (0.5 A/pix) j1238+1009
input_cube = "j1238+1009_main_icubes.fits"
err_cube   = "j1238+1009_main_vcubes.fits"
zspec = 0.003795
xc = 63  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 60

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

# Nite 2 (0.5 A/pix) j0823+0313
input_cube = "j0823+0313_17frames_icubes.fits"
err_cube   = "j0823+0313_17frames_vcubes.fits"
zspec = 0.009777
xc = 63  # spot (iraf units) to show in plot, which also shows the sum of all spatial pixels
yc = 60

inputlist.append(input_cube)
errlist.append(err_cube)
zsplist.append(zspec)
xclist.append(xc)
yclist.append(yc)

nzilist=[]
khilist=[]
klolist=[]
wvaclist=[]
wairlist=[]
hflist=[]
lflist=[]

# HeII 4686 FILTER -- keep it narrow
'''
nzi = 13  
khi = 24 
klo = 46 
wvac = 4686.99
wair = 4685.68
#flag_hi = True

nzilist.append(nzi)
khilist.append(khi)
klolist.append(klo)
wvaclist.append(wvac)
wairlist.append(wair)
hflist.append(True)
lflist.append(False)
'''

# [OII] image of z=0.275215 BG Galaxy (which messed up initial He II image).
#zspec = 0.275215   # Set redshift for BG galaxy image
#nzi = 13  
#khi = 24 
#klo = 46 
#wvac = 3728.76
#wair = 3727.70
#flag_lo = True


# Hb 4861 FILTER 
'''
nzi = 29
khi = 24 
klo = 46 
wvac = 4862.69
wair = 4861.33

nzilist.append(nzi)
khilist.append(khi)
klolist.append(klo)
wvaclist.append(wvac)
wairlist.append(wair)
hflist.append(False)
lflist.append(False)
'''

# [OIII] 4363 FILTER 
'''
nzi = 25
khi = 15
klo = 25
wvac = 4364.44
wair = 4363.21

nzilist.append(nzi)
khilist.append(khi)
klolist.append(klo)
wvaclist.append(wvac)
wairlist.append(wair)
hflist.append(False)
lflist.append(False)
'''

# [OIII] 5007 -- Avoid line at 5015.5776 (He I)
nzi = 29
khi = 35
klo = 30
wvac = 5008.24
wair = 5006.84  

nzilist.append(nzi)
khilist.append(khi)
klolist.append(klo)
wvaclist.append(wvac)
wairlist.append(wair)
hflist.append(False)
lflist.append(False)       

# [OIII] 4959 
#nzi = 29
#khi = 38
#klo = 46 
#wvac = 4960.30
#wair = 4958.92      

# [OIII] 4959 (J028-0817)
#nzi = 42
#khi = 38
#klo = 46 
#wvac = 4960.30
#wair = 4958.92        

# [OII] 3727 doublet
nzi = 15
khi = 38
klo = 46 
wvac = 3728.76
wair = 3727.70

nzilist.append(nzi)
khilist.append(khi)
klolist.append(klo)
wvaclist.append(wvac)
wairlist.append(wair)
hflist.append(False)
lflist.append(False) 


# HeI 4471
#nzi = 13
#khi = 35
#klo = 35
#wvac = 4472.729
#wair = 4471.474        

# [ArIV] 4740                                                                                             
#nzi = 13                                                                                                 
#khi = 35                                                                                                 
#klo = 35                                                                                                 
#wvac = 4741.53                                                                                           
#wair = 4740.20                                                                                           


# [ArIV] 4711                                                                                             
#nzi = 13                                                                                                 
#khi = 35                                                                                                 
#klo = 35                                                                                                 
#wvac = 4712.67                                                                                           
#wair = 4711.35                                                                                           


# [NeIII] 3868                                                                                            
#nzi = 13
#khi = 35
#klo = 35
#wvac = 3869.86
#wair = 3868.76



#################################
##### THE NB FILTER #############
#################################
# We compute the index of the center of the filter (zzpi-1) below.
# nzi (an odd integer) define the width of the filter in pixels (wavelength direction).
# Set the window for measuring the continuum
#  k is the index relative to the line center (integer).

for q in range(len(xclist)):
    for w in range(len(nzilist)):
        input_cube=inputlist[q]
        err_cube=errlist[q]
        zspec=zsplist[q]
        xc=xclist[q]
        yc=yclist[q]

        nzi=nzilist[w]
        khi=khilist[w]
        klo=klolist[w]
        wvac=wvaclist[w]
        wair=wairlist[w]
        flag_hi=hflist[w]
        flag_lo=lflist[w]

        # KCWI uses air wavelengths
        wrest = wair


        # ... slice off the 'fits' extension
        base = dest+'image.' + input_cube[:-5]  

        outfile = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '.fits'
        errfile = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '_E.fits'
        bgfile  = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '_CONT.fits'
        snfile  = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '_SNR.fits'
        varfile = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '_VAR.fits'
        bsigfile = base +  str(wrest)[:4] .zfill(4) + '_' +  str(nzi)  + '_BSIG.fits'

        hdulist = fits.open(source+input_cube)
        scidata = hdulist[0].data
        hdulist.close()

        hduerr = fits.open(source+err_cube)
        errdata = hduerr[0].data
        hduerr.close()

        #... Use a tuple to transfer HEADER KEYWORD and COMMENT at the same time
        crval1 = (hdulist[0].header['CRVAL1'], hdulist[0].header.comments['CRVAL1'])
        crpix1 = (hdulist[0].header['CRPIX1'], hdulist[0].header.comments['CRPIX1'])
        cd1_1  = (hdulist[0].header['CD1_1'],  hdulist[0].header.comments['CD1_1'])
        cname1 = (hdulist[0].header['CNAME1'], hdulist[0].header.comments['CNAME1'])
        ctype1 = (hdulist[0].header['CTYPE1'], hdulist[0].header.comments['CTYPE1'])
        cunit1 = (hdulist[0].header['CUNIT1'], hdulist[0].header.comments['CUNIT1'])
        #
        cd1_2 =  (hdulist[0].header['CD1_2'],  hdulist[0].header.comments['CD1_2'])
        cd2_1 =  (hdulist[0].header['CD2_1'],  hdulist[0].header.comments['CD2_1'])
        ttime =  (hdulist[0].header['TTIME'],  hdulist[0].header.comments['TTIME'])
        #
        crval2 = (hdulist[0].header['CRVAL2'], hdulist[0].header.comments['CRVAL2'])
        crpix2 = (hdulist[0].header['CRPIX2'], hdulist[0].header.comments['CRPIX2'])
        cd2_2  = (hdulist[0].header['CD2_2'],  hdulist[0].header.comments['CD2_2'])
        cname2 = (hdulist[0].header['CNAME2'], hdulist[0].header.comments['CNAME2'])
        ctype2 = (hdulist[0].header['CTYPE2'], hdulist[0].header.comments['CTYPE2'])
        cunit2 = (hdulist[0].header['CUNIT2'], hdulist[0].header.comments['CUNIT2'])
        #
        bunit = (hdulist[0].header['BUNIT'], hdulist[0].header.comments['BUNIT'])  # Flux Units

        #
        # Find the center of the NB filter; this is the wavelength pixel in ds9 units

        # Find the wavelength pixel in ds9 units
        wobs = wrest * (1. + zspec)
        crval3 = (hdulist[0].header['CRVAL3'], hdulist[0].header.comments['CRVAL3'])
        crpix3 = (hdulist[0].header['CRPIX3'], hdulist[0].header.comments['CRPIX3'])
        cd3_3  = (hdulist[0].header['CD3_3'],  hdulist[0].header.comments['CD3_3'])
        # this is the FITS (and iraf) index;  use [zzpi - 1] in python, which is zero-based
        zzpi = np.int( ((wobs - crval3[0]) / cd3_3[0] + crpix3[0]) + 0.5) - 1


        ### Choose the wavelength slice [All positions,Y,X]
        #image = scidata[zzpi-1,:,:]
        # initialize the images
        nxi = scidata[0,0,:].size
        nyi = scidata[0,:,0].size
        image = np.zeros((nyi,nxi))
        err_image = np.zeros((nyi,nxi))
        snr_image = np.zeros((nyi,nxi))
        var_image = np.zeros((nyi,nxi))
        bg = np.zeros((nyi,nxi))
        bghi = np.zeros((nyi,nxi))
        bglo = np.zeros((nyi,nxi))
        bsig = np.zeros((nyi,nxi))


        # Do the spectral extraction and sum for k in range(nzi):  zmin = -nzi/2+1, zmax = nzi/2+1
        k1  = -np.int(nzi/2+1)-klo + (zzpi-1)
        k1b = -np.int(nzi/2+1) + (zzpi-1)
        k2  = np.int(nzi/2+1)+khi + (zzpi-1)
        k2a = np.int(nzi/2+1) + (zzpi-1)

        w1  = crval3[0] + cd3_3[0] * (k1 - (crpix3[0] - 1))
        w2  = crval3[0] + cd3_3[0] * (k2 - (crpix3[0] - 1))
        w1b = crval3[0] + cd3_3[0] * (k1b - (crpix3[0] - 1))
        w2a = crval3[0] + cd3_3[0] * (k2a - (crpix3[0] - 1))

        if flag_hi:   # This removes the [OII] line from the J1044+0353 BG galaxy
            k2a_prime = 2856
            w2a_prime = crval3[0] + cd3_3[0] * (k2a_prime - (crpix3[0] - 1))
        else:
            k2a_prime = k2a
            w2a_prime = w2a

        #if flag_lo:   # This removes the J1044+0353 He II emission from the BGgal (z=0.275215) [OII] image
        #    k1b_prime = 2827
        #    w1b_prime = crval3[0] + cd3_3[0] * (k1b_prime - (crpix3[0] - 1))
        #else:
        #    k1b_prime = k1b
        #    w1b_prime = w1b
            
        if flag_lo:   
            k1b_prime = 2039
            w1b_prime = crval3[0] + cd3_3[0] * (k1b_prime - (crpix3[0] - 1))
        else:
            k1b_prime = k1b
            w1b_prime = w1b


        # Estimate background at (x,y) with continuum -klo and +khi beyond the extraction region
        # Note: k is the index relative to the pixel nearest the line center
        cnt = 0
        #for k in range(k2a, k2+1):  # include khi
        for k in range(k2a_prime, k2+1):  # include khi
            bghi = bghi + scidata[k,:,:]
            cnt = cnt + 1

        bghi = bghi / cnt




        cnt = 0
        #for k in range(k1,k1b+1):  #include the last point
        for k in range(k1,k1b_prime+1):  #include the last point
            bglo = bglo + scidata[k,:,:]
            cnt = cnt + 1

        bglo = bglo / cnt    

        bg = (bghi + bglo) / 2.0
        # Estimated stdv in background
        bsig = (bghi - bglo) / 2.0

        # Check the background estimate
        spec = np.sum(scidata,axis=(1,2))
        wave = np.zeros(spec.size)
        for k in range(spec.size):
            wave[k] = crval3[0] + cd3_3[0] * (k - (crpix3[0] - 1))

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(wave,spec, label="sum", alpha=0.5)

        fmax = spec[k1:k2].max()
        fmin = spec[k1:k2].min()
        print(fmin, fmax)
        # total background summed over all spatial (x,y).
        bgt   = np.sum(bg,axis=(0,1))
        bglot = np.sum(bglo,axis=(0,1))
        bghit = np.sum(bghi,axis=(0,1))
        plt.plot([w1,w2],[bgt,bgt], 'k', alpha=0.5)

        #plt.plot([w1,w1b],[bglot,bglot], 'k:', linewidth=5, alpha=0.5)
        plt.plot([w1,w1b_prime],[bglot,bglot], 'k:', linewidth=5, alpha=0.5)

        #plt.plot([w2a,w2],[bghit,bghit], 'k:', linewidth=5, alpha=0.5)
        plt.plot([w2a_prime,w2],[bghit,bghit], 'k:', linewidth=5, alpha=0.5)

        plt.plot([w1b,w2a],[bgt,bgt], 'r', linewidth=5, alpha=0.5)
        plt.axis([w1,w2,fmin,fmax*1.05])
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux/pix (1e-16)')
        title = "nb_extract " + str(input_cube)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(wave,scidata[:,yc-1,xc-1],label="(yc,xc)", alpha=0.5)  # convert to zero-based index from FITS one-based index
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux/pix (1e-16)')
        fmax = scidata[k1:k2,yc-1,xc-1].max()
        fmin = scidata[k1:k2,yc-1,xc-1].min()
        plt.axis([w1,w2,-fmin,fmax*1.05])
        plt.fill_between([w1b,w2a],[fmin,fmin],[fmax,fmax], alpha=0.5)
        plt.legend()
        plt.tight_layout()


        #fig.show()
        plt.close()


        # Integrate the line flux and variance
        #for k in range(np.int(-nzi/2+1),np.int(nzi/2+1)+1):
        for k in range(k1b+1,k2a):
            #print (k + (zzpi-1), scidata[k+(zzpi-1), yc, xc], bglo[yc,xc], bghi[yc,xc], scidata[k+(zzpi-1), yc, xc] - bg[yc,xc])
            #image = image + (scidata[k+(zzpi-1),:,:] - bg[:,:])
            print (k, scidata[k, yc, xc], bglo[yc,xc], bghi[yc,xc], scidata[k, yc, xc] - bg[yc,xc])
            image = image + (scidata[k,:,:] - bg[:,:])
            # add the variance at each image pixel
            err_image = err_image + errdata[k,:,:]

        var_image = np.copy(err_image)
        err_image = np.sqrt(err_image + bsig * bsig)
        snr_image = image * (err_image !=0) / (err_image + (err_image == 0))  # Set SNR=0 when error not defined
        #snr_image = image / err_image
        for j in range(nyi):
            for i in range(nxi):
                if err_image[j,i] == 0:
                    print("j=", j,  " i=", i, " err_image=", err_image[j,i], " bsig[j,i]=", bsig[j,i])


        # Create a PrimaryHDU object to encapsulate the data:
        hdu = fits.PrimaryHDU(image)
        # You could write this directlyto file
        # hdu.writeto('myspec_fancy.fits')
        # But we need to add header keywords first

        # Create a HDUList to contain the newlycreatedHDU object
        newhdulist = fits.HDUList([hdu])
        newhdulist[0].header['CRVAL1'] = crval1
        newhdulist[0].header['CRPIX1'] = crpix1
        newhdulist[0].header['CD1_1']  = cd1_1
        newhdulist[0].header['CDELT1'] = cd1_1
        newhdulist[0].header['CNAME1'] = cname1
        newhdulist[0].header['CTYPE1'] = ctype1
        newhdulist[0].header['CUNIT1'] = cunit1
        #
        newhdulist[0].header['CD1_2']  = cd1_2
        newhdulist[0].header['CD2_1']  = cd2_1
        newhdulist[0].header['TTIME']  = ttime
        #
        newhdulist[0].header['CRVAL2'] = crval2
        newhdulist[0].header['CRPIX2'] = crpix2
        newhdulist[0].header['CD2_2']  = cd2_2
        newhdulist[0].header['CDELT2'] = cd2_2
        newhdulist[0].header['CNAME2'] = cname2
        newhdulist[0].header['CTYPE2'] = ctype2
        newhdulist[0].header['CUNIT2'] = cunit2
        #
        newhdulist[0].header['BUNIT'] = bunit
        newhdulist[0].header['PHOTFLAM'] = 1e-16
        newhdulist[0].header.comments['PHOTFLAM'] = 'inverse sensitivity, ergs/cm2/Ang/count'
        newhdulist.writeto(outfile,overwrite=True)



        '''
        # Error Image
        hdu = fits.PrimaryHDU(err_image)
        newhdulist = fits.HDUList([hdu])
        newhdulist[0].header['CRVAL1'] = crval1
        newhdulist[0].header['CRPIX1'] = crpix1
        newhdulist[0].header['CD1_1']  = cd1_1
        newhdulist[0].header['CDELT1'] = cd1_1
        newhdulist[0].header['CNAME1'] = cname1
        newhdulist[0].header['CTYPE1'] = ctype1
        newhdulist[0].header['CUNIT1'] = cunit1
        #
        newhdulist[0].header['CD1_2']  = cd1_2
        newhdulist[0].header['CD2_1']  = cd2_1
        newhdulist[0].header['TTIME']  = ttime
        #
        newhdulist[0].header['CRVAL2'] = crval2
        newhdulist[0].header['CRPIX2'] = crpix2
        newhdulist[0].header['CD2_2']  = cd2_2
        newhdulist[0].header['CDELT2'] = cd2_2
        newhdulist[0].header['CNAME2'] = cname2
        newhdulist[0].header['CTYPE2'] = ctype2
        newhdulist[0].header['CUNIT2'] = cunit2
        #
        newhdulist[0].header['BUNIT'] = bunit
        newhdulist.writeto(errfile,overwrite=True)

        # Background Image
        hdu = fits.PrimaryHDU(bg)
        newhdulist = fits.HDUList([hdu])
        newhdulist[0].header['CRVAL1'] = crval1
        newhdulist[0].header['CRPIX1'] = crpix1
        newhdulist[0].header['CD1_1']  = cd1_1
        newhdulist[0].header['CDELT1'] = cd1_1
        newhdulist[0].header['CNAME1'] = cname1
        newhdulist[0].header['CTYPE1'] = ctype1
        newhdulist[0].header['CUNIT1'] = cunit1
        #
        newhdulist[0].header['CD1_2']  = cd1_2
        newhdulist[0].header['CD2_1']  = cd2_1
        newhdulist[0].header['TTIME']  = ttime
        #
        newhdulist[0].header['CRVAL2'] = crval2
        newhdulist[0].header['CRPIX2'] = crpix2
        newhdulist[0].header['CD2_2']  = cd2_2
        newhdulist[0].header['CDELT2'] = cd2_2
        newhdulist[0].header['CNAME2'] = cname2
        newhdulist[0].header['CTYPE2'] = ctype2
        newhdulist[0].header['CUNIT2'] = cunit2
        #
        newhdulist[0].header['BUNIT'] = bunit
        newhdulist.writeto(bgfile,overwrite=True)

        # SNR Image
        hdu = fits.PrimaryHDU(snr_image)
        newhdulist = fits.HDUList([hdu])
        newhdulist[0].header['CRVAL1'] = crval1
        newhdulist[0].header['CRPIX1'] = crpix1
        newhdulist[0].header['CD1_1']  = cd1_1
        newhdulist[0].header['CDELT1'] = cd1_1
        newhdulist[0].header['CNAME1'] = cname1
        newhdulist[0].header['CTYPE1'] = ctype1
        newhdulist[0].header['CUNIT1'] = cunit1
        #
        newhdulist[0].header['CD1_2']  = cd1_2
        newhdulist[0].header['CD2_1']  = cd2_1
        newhdulist[0].header['TTIME']  = ttime
        #
        newhdulist[0].header['CRVAL2'] = crval2
        newhdulist[0].header['CRPIX2'] = crpix2
        newhdulist[0].header['CD2_2']  = cd2_2
        newhdulist[0].header['CDELT2'] = cd2_2
        newhdulist[0].header['CNAME2'] = cname2
        newhdulist[0].header['CTYPE2'] = ctype2
        newhdulist[0].header['CUNIT2'] = cunit2
        #
        newhdulist[0].header['BUNIT'] = bunit
        newhdulist.writeto(snfile,overwrite=True)
        '''

        # Variance Image
        hdu = fits.PrimaryHDU(var_image)
        newhdulist = fits.HDUList([hdu])
        newhdulist[0].header['CRVAL1'] = crval1
        newhdulist[0].header['CRPIX1'] = crpix1
        newhdulist[0].header['CD1_1']  = cd1_1
        newhdulist[0].header['CDELT1'] = cd1_1
        newhdulist[0].header['CNAME1'] = cname1
        newhdulist[0].header['CTYPE1'] = ctype1
        newhdulist[0].header['CUNIT1'] = cunit1
        #
        newhdulist[0].header['CD1_2']  = cd1_2
        newhdulist[0].header['CD2_1']  = cd2_1
        newhdulist[0].header['TTIME']  = ttime
        #
        newhdulist[0].header['CRVAL2'] = crval2
        newhdulist[0].header['CRPIX2'] = crpix2
        newhdulist[0].header['CD2_2']  = cd2_2
        newhdulist[0].header['CDELT2'] = cd2_2
        newhdulist[0].header['CNAME2'] = cname2
        newhdulist[0].header['CTYPE2'] = ctype2
        newhdulist[0].header['CUNIT2'] = cunit2
        #
        newhdulist[0].header['BUNIT'] = bunit
        newhdulist.writeto(varfile,overwrite=True)

