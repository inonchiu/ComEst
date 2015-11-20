#!/usr/bin/env python

import numpy as np
from math import *
import matplotlib.pyplot as pyplt
import galsim
import comest
import sys
import matplotlib.pyplot as pyplt
import os
from distutils.spawn import find_executable
import pickle

#####
#
# Set the file names and configurations
#
#####

# get the current working directory
CWD             =       os.path.dirname( os.path.abspath(__name__) )

# get the sex path
sex_exec        =       find_executable("sex")

# get the file name
path2img        =       os.path.join( CWD, "DEEP2329+0012_i.fits" )
path2outdir     =       os.path.join( CWD, "outdir_example" )
sex_config      =       os.path.join( os.path.dirname( comest.__file__ ), "templates/sex.config" )
sex_params      =       os.path.join( os.path.dirname( comest.__file__ ), "templates/sex.params" )
full_root_name  =       "full"
bnb_root_name   =       "bnb"
full_sex_args   =       "-FILTER_NAME" + "    " +  os.path.join( os.path.dirname( comest.__file__ ), "templates/filters/gauss_3.0_5x5.conv" ) + "    " + \
                        "-DETECT_MINAREA 5   -DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 -DEBLEND_NTHRESH 32 -DEBLEND_MINCONT 0.005 -CLEAN Y -CLEAN_PARAM 1.0 -BACKPHOTO_THICK 24.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE" + " " + path2img + "[1]"
bnb_sex_args    =       "-FILTER_NAME" + "    " +  os.path.join( os.path.dirname( comest.__file__ ), "templates/filters/gauss_3.0_5x5.conv" ) + "    " + \
                        "-DETECT_MINAREA 250 -DETECT_THRESH 5.0 -ANALYSIS_THRESH 5.0 -DEBLEND_NTHRESH 32 -DEBLEND_MINCONT 0.005 -CLEAN Y -CLEAN_PARAM 1.0 -BACKPHOTO_THICK 24.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE" + " " + path2img + "[1]"
img_zp          =       31.7
img_pixel_scale =       0.26
img_fwhm        =       0.9     # from Desai+12 i band image.

MAG_LO          =       20.0
MAG_HI          =       27.0

nsimimages      =       10
ncpu            =       10
nset            =       10

# set the random seed
np.random.seed(232)

#####
#
# Now load the image
#
#####

# create an instance for image
fits_image_example  =   comest.ComEst.fitsimage(
                        path2img = path2img,
                        path2outdir = path2outdir,
                        sex_exec = sex_exec,
                        sex_config = sex_config,
                        sex_params = sex_params,
                        full_root_name = full_root_name,
                        bnb_root_name = bnb_root_name,
                        full_sex_args = full_sex_args,
                        bnb_sex_args = bnb_sex_args,
                        img_zp = img_zp,
                        img_pixel_scale = img_pixel_scale,
                        img_fwhm = img_fwhm)

# introduce yourself
fits_image_example.i_am()


#####
#
# Run SE to detect all sources
#
#####

# run and it will create a bunch of fits containing full srcs
fits_image_example.RunSEforAll()

# create the srcfree fits image and save it.
fits_image_example.SaveSrcFreeFits()


#####
#
# Run SE to detect BnB sources
#
#####

# run and it will create a bunch of fits containing BnB srcs
fits_image_example.RunSEforBnB()

# create the BnB fits image and save it.
fits_image_example.SaveBnBFits()

#####
#
# Galsim to put the srcs on the image
#
#####

for nsnset in xrange(nset):
    mef_buldisk, cat_buldisk = fits_image_example.BulDiskLocator(
                           path2image = fits_image_example.path2outdir + "/" + bnb_root_name + ".identical.bnb.fits",
                           nsimimages = nsimimages,
                           ngals_arcmin2 = 15.0,
                           psf_dict   = {"moffat":{ "beta": 3.5, "fwhm": img_fwhm} },
                           ncpu       = ncpu,
                           mag_dict   = {"lo":MAG_LO, "hi":MAG_HI },
                           random_seed= int( np.random.random() * 1000* 2.0**(nsnset + np.random.random()) ),
                           sims_nameroot = "buldisk_"+"%i" % nsnset)

for nsnset in xrange(nset):
    mef_pntsrcs, cat_pntsrcs = fits_image_example.PntSrcLocator(
                           path2image = fits_image_example.path2outdir + "/" + bnb_root_name + ".identical.bnb.fits",
                           nsimimages = nsimimages,
                           ngals_arcmin2 = 15.0,
                           psf_dict   = {"moffat":{ "beta": 3.5, "fwhm": img_fwhm} },
                           ncpu       = ncpu,
                           mag_dict   = {"lo":MAG_LO, "hi":MAG_HI },
                           random_seed= int( np.random.random() * 1000* 2.0**(nsnset + np.random.random()) ),
                           sims_nameroot = "pntsrc_"+"%i" % nsnset)

for nsnset in xrange(nset):
    mef_realgal, cat_realgal = fits_image_example.RealGalLocator(
                           path2image = fits_image_example.path2outdir + "/" + bnb_root_name + ".identical.bnb.fits",
                           nsimimages = nsimimages,
                           ngals_arcmin2 = 15.0,
                           psf_dict   = {"moffat":{ "beta": 3.5, "fwhm": img_fwhm} },
                           ncpu       = ncpu,
                           mag_dict   = {"lo":MAG_LO, "hi":MAG_HI },
                           random_seed= int( np.random.random() * 1000* 2.0**(nsnset + np.random.random()) ),
                           sims_nameroot = "realgal_"+"%i" % nsnset)

#####
#
# Run SE
#
#####

for nsnset in xrange(nset):
    fits_image_example.RunSEforSims( sims_nameroot = "buldisk_"+"%i" % nsnset, sims_sex_args = fits_image_example.full_sex_args, outputcheckimage = False, tol_fwhm=1.0, path2maskmap = fits_image_example.path2outdir + "/" + bnb_root_name + ".segmentation.fits")
    fits_image_example.RunSEforSims( sims_nameroot = "realgal_"+"%i" % nsnset, sims_sex_args = fits_image_example.full_sex_args, outputcheckimage = False, tol_fwhm=1.0, path2maskmap = fits_image_example.path2outdir + "/" + bnb_root_name + ".segmentation.fits")
    fits_image_example.RunSEforSims( sims_nameroot = "pntsrc_" +"%i" % nsnset, sims_sex_args = fits_image_example.full_sex_args, outputcheckimage = False, tol_fwhm=1.0, path2maskmap = fits_image_example.path2outdir + "/" + bnb_root_name + ".segmentation.fits")


#####
#
# Histgram
#
#####

HIST_DICT        =       {
    ttype + "_" + "%i" % nsnset   :   {
        "com"   :   fits_image_example.DeriveCom(sims_nameroot = ttype + "_" + "%i" % nsnset, x_steps_arcmin = 3.0, y_steps_arcmin = 3.0, mag_edges = np.arange(20.0,27.0,0.1), save_files = True),
        "pur"   :   fits_image_example.DerivePur(sims_nameroot = ttype + "_" + "%i" % nsnset, x_steps_arcmin = 3.0, y_steps_arcmin = 3.0, mag_edges = np.arange(20.0,27.0,0.1), save_files = True),
    } for ttype in ["buldisk", "realgal", "pntsrc"] for nsnset in xrange(nset)
}


# Save the pickle file
FF      =       open("output.pickle", "wb")
pickle.dump(HIST_DICT, FF)
FF.close()


