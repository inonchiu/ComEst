#!/usr/bin/env python

####################################################################################
#
# This script reads the input src free image and put fake galaxies and point sources on it.
# This script heavily relies on the demo provided in GalSim package.
#
####################################################################################
# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import sys
import os
from math import *
import numpy as np
import logging
import time
import galsim
import multiprocessing

#########################
#
# Functions
#
#########################


def _Flux2Mag(flux,ZP):
    """
    The conversion from Flux to magnitude defined in SE.
    The image has to have exposure time = 1 sec and
    the flux is the total counts (ADU or physical units) in the image.

    mag = -2.5 * log10(flux) + ZP
    Note: this function does not deal with nan.

    Parameters:
        -`flux`: nd array. the flux of the object.
        -`ZP`: float. the zeropoint.
    Return:
        -`mag`: nd array. the magnitude of the object.
    """
    return -2.5 * np.log10(flux) + ZP

def _Mag2Flux(mag, ZP):
    """
    The conversion from magnitude to flux defined in SE.
    The image has to have exposure time = 1 sec and
    the flux is the total counts (ADU or physical units) in the image.

    mag = -2.5 * log10(flux) + ZP
    Note: this function does not deal with nan.

    Parameters:
        -`magnitude`: nd array. the magnitude of the object.
        -`ZP`: float. the zeropoint.
    Return:
        -`flux`: nd array. the flux of the object.
    """
    return 10.0**((mag - ZP)/-2.5)

# ---
# Set up the worker
# ---
def _worker(targeted_routine, args_and_info, results_info_proc):
    """
    This is the generator worker routine for multiprocessing.

    Parameters:
        -`targeted_routine`: function object. The targeted function you want to run and pass the args_and_info to.
        -`args_and_info`: multiprocessing.Queue. It is a queue with (args, info) tuples.
                          args are the arguements to pass to targeted_routine
                          info is passed along to the output queue.
        -`results_info_proc`: multiprocessing.Queue.
                              It is a queue storing (result, info, time) tuples:
                              result is the return value of from targeted_routine
                              info is passed through from the input queue.
                              proc is the process name.
    """
    for (args, info) in iter(args_and_info.get, 'STOP'):
        result = targeted_routine(*args)
        results_info_proc.put( (result, info, multiprocessing.current_process().name) )


#########################
#
# Different functions to put the sources on the targeted image
#
#########################


def _PntSrcLocator_single(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    random_seed   = 8241573
    ):
    """
    This function reads the path2image and put the fake point sources on it.
    The properties of the point sources are just the gaussian convolved with the psf.
    There are some redundant configure parameters (e.g., the axis ratio) that we do not pass into the point sources, the reason
    for keeping this is just to be consistent with the galaxies configuartion.

    Parameters:
        -`path2image`: string. The abs path to the image which you want to put the fake galaxies. It is usually
                       the source free image if you want to estimate the completeness as a function of image. It
                       can also be the BnB image if you want to simulate the galaxies in the image with Bright and Big
                       sources.
        -`zeropoint`: float. The zeropoint of the image.
        -`psf_dict`: dict. The moffat psf configuration. It has to be in the form of {"moffat":{"lo": [value], "high": [value]}}.
        -`stamp_size_arcec`: float. The size (in arcmin) the GalSim will create for one single galaxy (or source).
        -`mag_dict`: dict. The magnitude configuration of GalSim galaxies in the unit of magnitude.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`hlr_dict`: dict. The half light radius configuration of GalSim galaxies in the unit of arcsec.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`fbulge_dict`: dict. The configuration of the fraction of the bulge component.
                     It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
        -`q_dict`: dict. The axis ratio configuration of GalSim galaxies.
                   It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
                   and q=1 means spherical.
        -`pos_ang_dict`: dict. The position angle configuration of GalSim galaxies in the unit of degree.
                         It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,180.0].
                         Moreover, it is counter-clockwise with +x is 0 degree.
        -`ngals_arcmin2`: float. The projected number per arcmin square of the galaxies you want to simulate.
        -`random_seed`: int. The random seed of the random generator.

    Returns:
        -`image`: galsim image. It is the simulated image outputed by Galsim.
        -`true_cats`: structured data array. It is the true catalog simulated by Galsim.

    """

    # ---
    # sanitize the parameters
    # ---
    # zeropoint
    zeropoint       =   float(zeropoint)
    # psf
    moffat_beta     =   float(psf_dict["moffat"]["beta"]) # the moffat psf beta, see https://www.gnu.org/software/gnuastro/manual/html_node/PSF.html
    moffat_fwhm     =   float(psf_dict["moffat"]["fwhm"]) # arcsec
    stamp_size_arcsec=  float(stamp_size_arcsec)          # arcsec.
    # galaxy and galaxy properties set up
    # galaxy master - magnitude / hlr / bulge fraction / axis_ratio / pos_ang
    # magnitude master
    mag_lo          =   float(mag_dict["lo"])         # the magnitude range
    mag_hi          =   float(mag_dict["hi"])         # tha magnitude range
    # hlr master
    hlr_lo          =   float(hlr_dict["lo"])         # arcsec
    hlr_hi          =   float(hlr_dict["hi"])         # arcsec
    # fbulge_master
    fbulge_lo       =   float(fbulge_dict["lo"])      # lower limit fraction of bulge in [0,1]
    fbulge_hi       =   float(fbulge_dict["hi"])      # upper limit fraction of bulge in [0,1]
    # axis_ratio
    q_lo            =   float(q_dict["lo"])           # the axis ratio min
    q_hi            =   float(q_dict["hi"])           # the axis ratio max
    # pos_ang
    pos_ang_lo      =   float(pos_ang_dict["lo"])     # deg, the lower limit of position angle in [0, 180.0]
    pos_ang_hi      =   float(pos_ang_dict["hi"])     # deg, the upper limit of position angle in [0, 180.0]
    # ngals density
    ngals_arcmin2   =   float(ngals_arcmin2)          # the density [arcmin^2] of the simualted gals

    # ---
    # Read in the image
    # ---
    # read in the full image and set up the properties of the full image
    full_image  =  galsim.fits.read(path2image)
    # get the pixel_scale from cd matrix -> there should be a simpler way but I dont know how right now.
    # it has to multiply 3600 to convert it from degree to arcsec.
    try:
        pixel_scale  =  sqrt( abs( np.linalg.det(full_image.wcs.cd) ) ) * 3600.0
    except:
        pixel_scale  =  sqrt( abs( galsim.fits.FitsHeader(path2image).header["CD1_1"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_2"] - \
                                   galsim.fits.FitsHeader(path2image).header["CD1_2"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_1"] ) ) * 3600.0
    # get the image size -> it seems it is 1-basis, hence the xmax means the number of pixel.
    full_image_xsize  =  full_image.xmax
    full_image_ysize  =  full_image.ymax

    # ---
    # ngals
    # ---
    # ngals
    ngals           =   np.int( full_image_xsize * full_image_ysize * ( pixel_scale / 60.0 )**2 * ngals_arcmin2 )  # the number of galaxies you want to simulate

    # ---
    # PSF properties
    # ---
    # Take the Moffat psf
    # Note: You may omit the flux, since the default is flux=1.
    psf = galsim.Moffat(flux = 1.0, beta = moffat_beta, fwhm = moffat_fwhm)
    # Take the (e1, e2) shape parameters for psf
    psf = psf.shear(e1=0.0, e2=0.0)

    # ---
    # Set the stamp size
    # ---
    # set up the stamp and stamp properties
    # The stamp size in pixel. it has to be the multiply of 2.
    stamp_size   =  np.int(stamp_size_arcsec  / pixel_scale)     \
    if  np.int(stamp_size_arcsec  / pixel_scale) % 2 == 0 else \
        np.int(stamp_size_arcsec  / pixel_scale) + 1
        #np.int(10.0 / pixel_scale) + 1


    # copy the image
    image_copy      =   full_image.copy()

    # true_cats
    # true data type
    data_type_in_cat=   np.dtype([ ("x_true"     , "f8"), ("y_true"          , "f8"),
                                   ("mag_true"   , "f8"), ("flux_true"       , "f8") ])
    # the catalog order is x, y, mag, flux, hlr(arcsec), fbulge, q, pos_ang
    true_cats       =   np.array([], dtype = data_type_in_cat)

    # take the random generator first
    # rng is the random number generator generated from the Deviation with the random_seed.
    # ud is the uniform random number between 0 and 1.
    rng = galsim.BaseDeviate(random_seed)
    ud  = galsim.UniformDeviate(rng)

    # ---
    # Start simulate the image
    # ---
    # looping all pnt
    for k in xrange(0, ngals):

        # generate the position of the stamp in the full image.
        xcen    =   ud() * full_image_xsize
        ycen    =   ud() * full_image_ysize
        # the pixel of gal center in the full image
        ixcen   =   int(floor(xcen+1.0))
        iycen   =   int(floor(ycen+1.0))
        # stamp size
        stamp_bounds = galsim.BoundsI(ixcen-0.5*stamp_size, ixcen+0.5*stamp_size-1,
                                      iycen-0.5*stamp_size, iycen+0.5*stamp_size-1)

        # pnt src properties
        pnt_mag     =   mag_lo + ud() * (mag_hi - mag_lo)

        # Point source properties
        # set the flue
        pnt_flux    =   _Mag2Flux(mag = pnt_mag, ZP = zeropoint)
        final = psf.withFlux(pnt_flux)

        # stamp the final gal image
        stamp = final.drawImage(bounds = stamp_bounds, scale = pixel_scale)

        # calc the overlapping bounds
        overlapping_bounds          =   stamp_bounds & full_image.bounds
        # add it to the full image
        image_copy[overlapping_bounds]    =   image_copy[overlapping_bounds] + stamp[overlapping_bounds]

        # collect pnt properties
        pnt_prop    =   np.array([ ( float(xcen)    , float(ycen)        ,
                                     float(pnt_mag) , float(pnt_flux) ) ],
                                   dtype = data_type_in_cat )

        # append gals in this image
        true_cats   =   np.append(true_cats, pnt_prop)

    # return
    return image_copy.copy(), true_cats.copy()


def _BulDiskLocator_single(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    random_seed   = 8241573
    ):
    """
    This function reads the path2image and put the fake galaxies on it.
    The properties of the fake galaxies are configured by several input parameters and they
    are convolved with psf before being put on the image.

    Parameters:
        -`path2image`: string. The abs path to the image which you want to put the fake galaxies. It is usually
                       the source free image if you want to estimate the completeness as a function of image. It
                       can also be the BnB image if you want to simulate the galaxies in the image with Bright and Big
                       sources.
        -`zeropoint`: float. The zeropoint of the image.
        -`psf_dict`: dict. The moffat psf configuration. It has to be in the form of {"moffat":{"lo": [value], "high": [value]}}.
        -`stamp_size_arcec`: float. The size (in arcmin) the GalSim will create for one single galaxy (or source).
        -`mag_dict`: dict. The magnitude configuration of GalSim galaxies in the unit of magnitude.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`hlr_dict`: dict. The half light radius configuration of GalSim galaxies in the unit of arcsec.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`fbulge_dict`: dict. The configuration of the fraction of the bulge component.
                     It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
        -`q_dict`: dict. The axis ratio configuration of GalSim galaxies.
                   It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
                   and q=1 means spherical.
        -`pos_ang_dict`: dict. The position angle configuration of GalSim galaxies in the unit of degree.
                         It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,180.0].
                         Moreover, it is counter-clockwise with +x is 0 degree.
        -`ngals_arcmin2`: float. The projected number per arcmin square of the galaxies you want to simulate.
        -`random_seed`: int. The random seed of the random generator.

    Returns:
        -`image`: galsim image. It is the simulated image outputed by Galsim.
        -`true_cats`: structured data array. It is the true catalog simulated by Galsim.

    """

    # ---
    # sanitize the parameters
    # ---
    # zeropoint
    zeropoint       =   float(zeropoint)
    # psf
    moffat_beta     =   float(psf_dict["moffat"]["beta"]) # the moffat psf beta, see https://www.gnu.org/software/gnuastro/manual/html_node/PSF.html
    moffat_fwhm     =   float(psf_dict["moffat"]["fwhm"]) # arcsec
    stamp_size_arcsec=  float(stamp_size_arcsec)          # arcsec.
    # galaxy and galaxy properties set up
    # galaxy master - magnitude / hlr / bulge fraction / axis_ratio / pos_ang
    # magnitude master
    mag_lo          =   float(mag_dict["lo"])         # the magnitude range
    mag_hi          =   float(mag_dict["hi"])         # tha magnitude range
    # hlr master
    hlr_lo          =   float(hlr_dict["lo"])         # arcsec
    hlr_hi          =   float(hlr_dict["hi"])         # arcsec
    # fbulge_master
    fbulge_lo       =   float(fbulge_dict["lo"])      # lower limit fraction of bulge in [0,1]
    fbulge_hi       =   float(fbulge_dict["hi"])      # upper limit fraction of bulge in [0,1]
    # axis_ratio
    q_lo            =   float(q_dict["lo"])           # the axis ratio min
    q_hi            =   float(q_dict["hi"])           # the axis ratio max
    # pos_ang
    pos_ang_lo      =   float(pos_ang_dict["lo"])     # deg, the lower limit of position angle in [0, 180.0]
    pos_ang_hi      =   float(pos_ang_dict["hi"])     # deg, the upper limit of position angle in [0, 180.0]
    # ngals density
    ngals_arcmin2   =   float(ngals_arcmin2)          # the density [arcmin^2] of the simualted gals

    # ---
    # Read in the image
    # ---
    # read in the full image and set up the properties of the full image
    full_image  =  galsim.fits.read(path2image)
    # get the pixel_scale from cd matrix -> there should be a simpler way but I dont know how right now.
    # it has to multiply 3600 to convert it from degree to arcsec.
    try:
        pixel_scale  =  sqrt( abs( np.linalg.det(full_image.wcs.cd) ) ) * 3600.0
    except:
        pixel_scale  =  sqrt( abs( galsim.fits.FitsHeader(path2image).header["CD1_1"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_2"] - \
                                   galsim.fits.FitsHeader(path2image).header["CD1_2"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_1"] ) ) * 3600.0
    # get the image size -> it seems it is 1-basis, hence the xmax means the number of pixel.
    full_image_xsize  =  full_image.xmax
    full_image_ysize  =  full_image.ymax

    # ---
    # ngals
    # ---
    # ngals
    ngals           =   np.int( full_image_xsize * full_image_ysize * ( pixel_scale / 60.0 )**2 * ngals_arcmin2 )  # the number of galaxies you want to simulate

    # ---
    # PSF properties
    # ---
    # Take the Moffat psf
    # Note: You may omit the flux, since the default is flux=1.
    psf = galsim.Moffat(flux = 1.0, beta = moffat_beta, fwhm = moffat_fwhm)
    # Take the (e1, e2) shape parameters for psf
    psf = psf.shear(e1=0.0, e2=0.0)

    # ---
    # Set the stamp size
    # ---
    # set up the stamp and stamp properties
    # The stamp size in pixel. it has to be the multiply of 2.
    stamp_size   =  np.int(stamp_size_arcsec  / pixel_scale)     \
    if  np.int(stamp_size_arcsec  / pixel_scale) % 2 == 0 else \
        np.int(stamp_size_arcsec  / pixel_scale) + 1
        #np.int(10.0 / pixel_scale) + 1


    # copy the image
    image_copy      =   full_image.copy()

    # true_cats
    # true data type
    data_type_in_cat=   np.dtype([ ("x_true"   , "f8"), ("y_true"          , "f8"), ("mag_true"   , "f8"),
                                   ("flux_true", "f8"), ("hlr_true[arcsec]", "f8"), ("fbulge_true", "f8"),
                                   ("q_true"   , "f8"), ("pos_ang[deg]"    , "f8") ])
    # the catalog order is x, y, mag, flux, hlr(arcsec), fbulge, q, pos_ang
    true_cats       =   np.array([], dtype = data_type_in_cat)

    # take the random generator first
    # rng is the random number generator generated from the Deviation with the random_seed.
    # ud is the uniform random number between 0 and 1.
    rng = galsim.BaseDeviate(random_seed)
    ud  = galsim.UniformDeviate(rng)

    # ---
    # Start simulate the image
    # ---

    # looping all gals
    for k in xrange(0, ngals):

        # generate the position of the stamp in the full image.
        xcen    =   ud() * full_image_xsize
        ycen    =   ud() * full_image_ysize
        # the pixel of gal center in the full image
        ixcen   =   int(floor(xcen+1.0))
        iycen   =   int(floor(ycen+1.0))
        # stamp size
        stamp_bounds = galsim.BoundsI(ixcen-0.5*stamp_size, ixcen+0.5*stamp_size-1,
                                      iycen-0.5*stamp_size, iycen+0.5*stamp_size-1)

        # gal properties
        gal_mag     =   mag_lo + ud() * (mag_hi - mag_lo)
        gal_hlr     =   hlr_lo + ud() * (hlr_hi - hlr_lo)
        gal_fbulge  =   fbulge_lo + ud() * (fbulge_hi - fbulge_lo)
        gal_q       =   q_lo   + ud() * (q_hi   - q_lo  )
        gal_pos_ang =   pos_ang_lo + ud() * (pos_ang_hi - pos_ang_lo )
        # Gal profile
        # Galaxy is a bulge + disk with parameters taken from the catalog:
        disk  = galsim.Exponential(flux               = (1.0 - gal_fbulge),
                                   half_light_radius  = gal_hlr             )
        #bulge = galsim.DeVaucouleurs(flux             = gal_fbulge,
        #                           half_light_radius  = gal_hlr,
        #                           flux_untruncated   = False )
        bulge  = galsim.Gaussian( flux               = gal_fbulge,
                                  half_light_radius  = gal_hlr )
        # sum disk and bulge
        gal = disk + bulge
        # set the flue
        gal_flux    =   _Mag2Flux(mag = gal_mag, ZP = zeropoint)
        gal = gal.withFlux(gal_flux)
        # set the ellipticity (or shear)
        gal = gal.shear(q = gal_q, beta = gal_pos_ang * galsim.degrees)
        # Convolve with psf
        final = galsim.Convolve([psf, gal])

        # stamp the final gal image
        stamp = final.drawImage(bounds = stamp_bounds, scale = pixel_scale)

        # calc the overlapping bounds
        overlapping_bounds          =   stamp_bounds & full_image.bounds
        # add it to the full image
        image_copy[overlapping_bounds]    =   image_copy[overlapping_bounds] + stamp[overlapping_bounds]

        # collect gal properties
        gal_prop    =   np.array([ ( float(xcen)    , float(ycen)        , float(gal_mag)   ,
                                     float(gal_flux), float(gal_hlr)     , float(gal_fbulge),
                                     float(gal_q)   , float(gal_pos_ang) ) ],
                                   dtype = data_type_in_cat )

        # append gals in this image
        true_cats   =   np.append(true_cats, gal_prop)

    # return
    return image_copy.copy(), true_cats.copy()


def _ModelGalLocator_single(
    path2image,
    readincat,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    random_seed   = 8241573
    ):
    """
    This function reads the path2image and put the fake galaxies on it.
    The properties of the fake galaxies are controlled by the input catalog and they
    are convolved with psf before being put on the image.

    Parameters:
        -`path2image`: string. The abs path to the image which you want to put the fake galaxies. It is usually
                       the source free image if you want to estimate the completeness as a function of image. It
                       can also be the BnB image if you want to simulate the galaxies in the image with Bright and Big
                       sources.
        -`readincat`: ndarray. This array consists of the input catalog, including the input values with the field names of
                      ``x_true``, ``y_true``, ``mag_true``, ``hlr_true[arcsec]``, ``fbulge_true``, ``q_true`` and ``pos_ang[deg]``.
        -`zeropoint`: float. The zeropoint of the image.
        -`psf_dict`: dict. The moffat psf configuration. It has to be in the form of {"moffat":{"lo": [value], "high": [value]}}.
        -`stamp_size_arcec`: float. The size (in arcmin) the GalSim will create for one single galaxy (or source).
        -`mag_dict`: dict. The magnitude configuration of GalSim galaxies in the unit of magnitude.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`hlr_dict`: dict. The half light radius configuration of GalSim galaxies in the unit of arcsec.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`fbulge_dict`: dict. The configuration of the fraction of the bulge component.
                     It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
        -`q_dict`: dict. The axis ratio configuration of GalSim galaxies.
                   It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
                   and q=1 means spherical.
        -`pos_ang_dict`: dict. The position angle configuration of GalSim galaxies in the unit of degree.
                         It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,180.0].
                         Moreover, it is counter-clockwise with +x is 0 degree.
        -`ngals_arcmin2`: float. The projected number per arcmin square of the galaxies you want to simulate.
        -`random_seed`: int. The random seed of the random generator.

    Returns:
        -`image`: galsim image. It is the simulated image outputed by Galsim.

    """

    # ---
    # sanitize the parameters
    # ---
    # zeropoint
    zeropoint       =   float(zeropoint)
    # psf
    moffat_beta     =   float(psf_dict["moffat"]["beta"]) # the moffat psf beta, see https://www.gnu.org/software/gnuastro/manual/html_node/PSF.html
    moffat_fwhm     =   float(psf_dict["moffat"]["fwhm"]) # arcsec
    stamp_size_arcsec=  float(stamp_size_arcsec)          # arcsec.
    # galaxy and galaxy properties set up
    # galaxy master - magnitude / hlr / bulge fraction / axis_ratio / pos_ang
    # magnitude master
    mag_lo          =   float(mag_dict["lo"])         # the magnitude range
    mag_hi          =   float(mag_dict["hi"])         # tha magnitude range
    # hlr master
    hlr_lo          =   float(hlr_dict["lo"])         # arcsec
    hlr_hi          =   float(hlr_dict["hi"])         # arcsec
    # fbulge_master
    fbulge_lo       =   float(fbulge_dict["lo"])      # lower limit fraction of bulge in [0,1]
    fbulge_hi       =   float(fbulge_dict["hi"])      # upper limit fraction of bulge in [0,1]
    # axis_ratio
    q_lo            =   float(q_dict["lo"])           # the axis ratio min
    q_hi            =   float(q_dict["hi"])           # the axis ratio max
    # pos_ang
    pos_ang_lo      =   float(pos_ang_dict["lo"])     # deg, the lower limit of position angle in [0, 180.0]
    pos_ang_hi      =   float(pos_ang_dict["hi"])     # deg, the upper limit of position angle in [0, 180.0]
    # ngals density
    ngals_arcmin2   =   float(ngals_arcmin2)          # the density [arcmin^2] of the simualted gals

    # ---
    # Read in the image
    # ---
    # read in the full image and set up the properties of the full image
    full_image  =  galsim.fits.read(path2image)
    # get the pixel_scale from cd matrix -> there should be a simpler way but I dont know how right now.
    # it has to multiply 3600 to convert it from degree to arcsec.
    try:
        pixel_scale  =  sqrt( abs( np.linalg.det(full_image.wcs.cd) ) ) * 3600.0
    except:
        pixel_scale  =  sqrt( abs( galsim.fits.FitsHeader(path2image).header["CD1_1"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_2"] - \
                                   galsim.fits.FitsHeader(path2image).header["CD1_2"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_1"] ) ) * 3600.0
    # get the image size -> it seems it is 1-basis, hence the xmax means the number of pixel.
    full_image_xsize  =  full_image.xmax
    full_image_ysize  =  full_image.ymax

    # ---
    # ngals
    # ---
    # ngals
    ngals           =   len(readincat)

    # ---
    # PSF properties
    # ---
    # Take the Moffat psf
    # Note: You may omit the flux, since the default is flux=1.
    psf = galsim.Moffat(flux = 1.0, beta = moffat_beta, fwhm = moffat_fwhm)
    # Take the (e1, e2) shape parameters for psf
    psf = psf.shear(e1=0.0, e2=0.0)

    # ---
    # Set the stamp size
    # ---
    # set up the stamp and stamp properties
    # The stamp size in pixel. it has to be the multiply of 2.
    stamp_size   =  np.int(stamp_size_arcsec  / pixel_scale)     \
    if  np.int(stamp_size_arcsec  / pixel_scale) % 2 == 0 else \
        np.int(stamp_size_arcsec  / pixel_scale) + 1
        #np.int(10.0 / pixel_scale) + 1

    # copy the image
    image_copy      =   full_image.copy()

    # true_cats
    x_true          =   readincat["x_true"          ].astype(np.float)
    y_true          =   readincat["y_true"          ].astype(np.float)
    mag_true        =   readincat["mag_true"        ].astype(np.float)
    hlr_true_arcsec =   readincat["hlr_true[arcsec]"].astype(np.float)
    fbulge_true     =   readincat["fbulge_true"     ].astype(np.float)
    q_true          =   readincat["q_true"          ].astype(np.float)
    pos_ang         =   readincat["pos_ang[deg]"    ].astype(np.float)

    # take the random generator first
    # rng is the random number generator generated from the Deviation with the random_seed.
    # ud is the uniform random number between 0 and 1.
    rng = galsim.BaseDeviate(random_seed)
    ud  = galsim.UniformDeviate(rng)

    # ---
    # Start simulate the image
    # ---

    # looping all gals
    for k in xrange(0, ngals):

        # generate the position of the stamp in the full image.
        xcen    =   x_true[k]
        ycen    =   y_true[k]
        # the pixel of gal center in the full image
        #ixcen   =   int(floor(xcen+0.5))
        #iycen   =   int(floor(ycen+0.5))
        ixcen   =   int(floor(xcen+1.0))
        iycen   =   int(floor(ycen+1.0))
        # stamp size
        stamp_bounds = galsim.BoundsI(ixcen-0.5*stamp_size, ixcen+0.5*stamp_size-1,
                                      iycen-0.5*stamp_size, iycen+0.5*stamp_size-1)

        # gal properties
        gal_mag     =   mag_true[k]
        gal_hlr     =   hlr_true_arcsec[k]
        gal_fbulge  =   fbulge_true[k]
        gal_q       =   q_true[k]
        gal_pos_ang =   pos_ang[k]

        # Gal profile
        # Galaxy is a bulge + disk with parameters taken from the catalog:
        disk  = galsim.Exponential(flux               = (1.0 - gal_fbulge),
                                   half_light_radius  = gal_hlr             )
        #bulge = galsim.DeVaucouleurs(flux             = gal_fbulge,
        #                           half_light_radius  = gal_hlr,
        #                           flux_untruncated   = False )
        bulge  = galsim.Gaussian( flux               = gal_fbulge,
                                  half_light_radius  = gal_hlr )
        # sum disk and bulge
        gal = disk + bulge
        # set the flue
        gal_flux    =   _Mag2Flux(mag = gal_mag, ZP = zeropoint)
        gal = gal.withFlux(gal_flux)
        # set the ellipticity (or shear)
        gal = gal.shear(q = gal_q, beta = gal_pos_ang * galsim.degrees)
        # Convolve with psf
        final = galsim.Convolve([psf, gal])

        # stamp the final gal image
        stamp = final.drawImage(bounds = stamp_bounds, scale = pixel_scale)

        # calc the overlapping bounds
        overlapping_bounds          =   stamp_bounds & full_image.bounds
        # add it to the full image
        image_copy[overlapping_bounds]    =   image_copy[overlapping_bounds] + stamp[overlapping_bounds]

    # return
    return image_copy.copy()


def _RealGalLocator_single(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    random_seed   = 8241573
    ):
    """
    This function reads the path2image and put the fake galaxies on it.
    The properties of the fake galaxies are directly taken from COSMOS catalog provided by Galsim team and they
    are convolved with psf before being put on the image.
    In this routine, we only use pos_ang_dict to re-rotate the galaxies.
    Please run `galsim_download_cosmos` provided by GalSim to download the COSMOS catalog - perhaps we should add the input catalog arguments.

    Parameters:
        -`path2image`: string. The abs path to the image which you want to put the fake galaxies. It is usually
                       the source free image if you want to estimate the completeness as a function of image. It
                       can also be the BnB image if you want to simulate the galaxies in the image with Bright and Big
                       sources.
        -`zeropoint`: float. The zeropoint of the image.
        -`psf_dict`: dict. The moffat psf configuration. It has to be in the form of {"moffat":{"lo": [value], "high": [value]}}.
        -`stamp_size_arcec`: float. The size (in arcmin) the GalSim will create for one single galaxy (or source).
        -`mag_dict`: dict. The magnitude configuration of GalSim galaxies in the unit of magnitude.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`hlr_dict`: dict. The half light radius configuration of GalSim galaxies in the unit of arcsec.
                     It has to be in the form of {"lo": [value], "high": [value]}.
        -`fbulge_dict`: dict. The configuration of the fraction of the bulge component.
                     It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
        -`q_dict`: dict. The axis ratio configuration of GalSim galaxies.
                   It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,1]
                   and q=1 means spherical.
        -`pos_ang_dict`: dict. The position angle configuration of GalSim galaxies in the unit of degree.
                         It has to be in the form of {"lo": [value], "high": [value]}. Note that the value has to be within [0,180.0].
                         Moreover, it is counter-clockwise with +x is 0 degree.
        -`ngals_arcmin2`: float. The projected number per arcmin square of the galaxies you want to simulate.
        -`random_seed`: int. The random seed of the random generator.

    Returns:
        -`image`: galsim image. It is the simulated image outputed by Galsim.
        -`true_cats`: structured data array. It is the true catalog simulated by Galsim.

    """

    # ---
    # sanitize the parameters
    # ---
    # zeropoint
    zeropoint       =   float(zeropoint)
    # psf
    moffat_beta     =   float(psf_dict["moffat"]["beta"]) # the moffat psf beta, see https://www.gnu.org/software/gnuastro/manual/html_node/PSF.html
    moffat_fwhm     =   float(psf_dict["moffat"]["fwhm"]) # arcsec
    stamp_size_arcsec=  float(stamp_size_arcsec)          # arcsec.
    # galaxy and galaxy properties set up
    # galaxy master - magnitude / hlr / bulge fraction / axis_ratio / pos_ang
    # magnitude master
    mag_lo          =   float(mag_dict["lo"])         # the magnitude range
    mag_hi          =   float(mag_dict["hi"])         # tha magnitude range
    # hlr master
    hlr_lo          =   float(hlr_dict["lo"])         # arcsec
    hlr_hi          =   float(hlr_dict["hi"])         # arcsec
    # fbulge_master
    fbulge_lo       =   float(fbulge_dict["lo"])      # lower limit fraction of bulge in [0,1]
    fbulge_hi       =   float(fbulge_dict["hi"])      # upper limit fraction of bulge in [0,1]
    # axis_ratio
    q_lo            =   float(q_dict["lo"])           # the axis ratio min
    q_hi            =   float(q_dict["hi"])           # the axis ratio max
    # pos_ang
    pos_ang_lo      =   float(pos_ang_dict["lo"])     # deg, the lower limit of position angle in [0, 180.0]
    pos_ang_hi      =   float(pos_ang_dict["hi"])     # deg, the upper limit of position angle in [0, 180.0]
    # ngals density
    ngals_arcmin2   =   float(ngals_arcmin2)          # the density [arcmin^2] of the simualted gals

    # ---
    # Read in the image
    # ---
    # read in the full image and set up the properties of the full image
    full_image  =  galsim.fits.read(path2image)
    # get the pixel_scale from cd matrix -> there should be a simpler way but I dont know how right now.
    # it has to multiply 3600 to convert it from degree to arcsec.
    try:
        pixel_scale  =  sqrt( abs( np.linalg.det(full_image.wcs.cd) ) ) * 3600.0
    except:
        pixel_scale  =  sqrt( abs( galsim.fits.FitsHeader(path2image).header["CD1_1"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_2"] - \
                                   galsim.fits.FitsHeader(path2image).header["CD1_2"] * \
                                   galsim.fits.FitsHeader(path2image).header["CD2_1"] ) ) * 3600.0
    # get the image size -> it seems it is 1-basis, hence the xmax means the number of pixel.
    full_image_xsize  =  full_image.xmax
    full_image_ysize  =  full_image.ymax

    # ---
    # ngals
    # ---
    # ngals
    ngals           =   np.int( full_image_xsize * full_image_ysize * ( pixel_scale / 60.0 )**2 * ngals_arcmin2 )  # the number of galaxies you want to simulate

    # ---
    # PSF properties
    # ---
    # Take the Moffat psf
    # Note: You may omit the flux, since the default is flux=1.
    psf = galsim.Moffat(flux = 1.0, beta = moffat_beta, fwhm = moffat_fwhm)
    # Take the (e1, e2) shape parameters for psf
    psf = psf.shear(e1=0.0, e2=0.0)

    # ---
    # Set the stamp size
    # ---
    # set up the stamp and stamp properties
    # The stamp size in pixel. it has to be the multiply of 2.
    stamp_size   =  np.int(stamp_size_arcsec  / pixel_scale)     \
    if  np.int(stamp_size_arcsec  / pixel_scale) % 2 == 0 else \
        np.int(stamp_size_arcsec  / pixel_scale) + 1
        #np.int(10.0 / pixel_scale) + 1


    # copy the image
    image_copy      =   full_image.copy()

    # true_cats
    # true data type
    data_type_in_cat=   np.dtype([ ("x_true"   , "f8"), ("y_true"          , "f8"), ("mag_true"   , "f8"),
                                   ("flux_true", "f8"), ("pos_ang[deg]"    , "f8") ])
    # the catalog order is x, y, mag, flux, pos_ang
    true_cats       =   np.array([], dtype = data_type_in_cat)

    # take the random generator first
    # rng is the random number generator generated from the Deviation with the random_seed.
    # ud is the uniform random number between 0 and 1.
    rng = galsim.BaseDeviate(random_seed)
    ud  = galsim.UniformDeviate(rng)


    # read in real galaxy catalog
    real_galaxy_catalog = galsim.RealGalaxyCatalog()

    # ---
    # Start simulate the image
    # ---

    # looping all gals
    for k in xrange(0, ngals):

        # generate the position of the stamp in the full image.
        xcen    =   ud() * full_image_xsize
        ycen    =   ud() * full_image_ysize
        # the pixel of gal center in the full image
        ixcen   =   int(floor(xcen+1.0))
        iycen   =   int(floor(ycen+1.0))
        # stamp size
        stamp_bounds = galsim.BoundsI(ixcen-0.5*stamp_size, ixcen+0.5*stamp_size-1,
                                      iycen-0.5*stamp_size, iycen+0.5*stamp_size-1)

        # gal properties
        gal_mag     =   mag_lo + ud() * (mag_hi - mag_lo)
        gal_pos_ang =   pos_ang_lo + ud() * (pos_ang_hi - pos_ang_lo )
        # Gal profile
        # Galaxy is a bulge + disk with parameters taken from the catalog:
        gal         =   galsim.RealGalaxy(real_galaxy_catalog, random = True)
        # set the flue
        gal_flux    =   _Mag2Flux(mag = gal_mag, ZP = zeropoint)
        gal = gal.withFlux(gal_flux)
        # set the rotation 
        gal = gal.rotate(gal_pos_ang * galsim.degrees)
        # Convolve with psf
        final = galsim.Convolve([psf, gal])

        # stamp the final gal image
        stamp = final.drawImage(bounds = stamp_bounds, scale = pixel_scale)

        # calc the overlapping bounds
        overlapping_bounds          =   stamp_bounds & full_image.bounds
        # add it to the full image
        image_copy[overlapping_bounds]    =   image_copy[overlapping_bounds] + stamp[overlapping_bounds]

        # collect gal properties
        gal_prop    =   np.array([ ( float(xcen)    , float(ycen)        , float(gal_mag)   ,
                                     float(gal_flux), float(gal_pos_ang) ) ],
                                   dtype = data_type_in_cat )

        # append gals in this image
        true_cats   =   np.append(true_cats, gal_prop)

    # return
    return image_copy.copy(), true_cats.copy()


#########################
#
# Engine for multiprocessing...
#
#########################


def BulDiskLocator(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    nsimimages    = 10,
    random_seed   = 8241573,
    ncpu          = 2,
    ):
    """

    :param path2image: The absolute path to the image which you want to put the simulated sources on. This is is usually the source free image (SFI), or it can also be the BnB image if you want to simulate the sources on the image where the observed BnB sources are kept. One can uses BnB image to test how the BnB sources affect the detection.

    :param psf_dict: The psf configuration. Currently it only supports Moffat PSF with beta parameter of 4.5. ``psf_dict`` must be a dictionary in the form of ``{"moffat":{ "beta": _value_, "fwhm": _value_ } }``, where _value_ of ``fwhm`` is in the unit of arcsec. By default, ``psf_dict = {"moffat":{ "beta": 4.5, "fwhm": img_fwhm } }``.

    :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 20.0``.

    :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.

    :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.

    :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.

    :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.

    :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.

    :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 25.0``.

    :param nsimimages: The number of the images you want to simulate. It will be saved in the multi-extension file with the code name ``sims_nameroot``. By default, it is ``nsimimages = 50``.

    :param random_seed: The random seed of the random generator. It will be passed to **GalSim** for simulating the sources.

    :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**. By default, it is ``sims_nameroot = "buldisk"``.

    :param ncpu: The number of cpu for parallel running. By default, it is ``ncpu = 2``. Please do not set this number higher than the CPU cores you have.

    :type path2image: str
    :type psf_dict: dict
    :type stamp_size_arcsec: float
    :type mag_dict: dict
    :type hlr_dict: dict
    :type fbulge_dict: dict
    :type q_dict: dict
    :type pos_ang_dict: dict
    :type ngals_arcmin2: float
    :type nsimimages: int
    :type random_seed: int
    :type sims_nameroot: str
    :type ncpu: int

    :returns: ``image_collection`` is the list containing the simulated image and ``true_cats_collection`` is the list containing the information of the mock catalog (hence ``len(image_collection) = len(true_cats_collection) = nsimimages``). Each element of the lists is the simulated image outputed by Galsim.
    :rtype: list, list

    This function reads the ``path2image`` and put the fake galaxies on it. In this case it is the galaxies consisting of buldge and disk components. The properties of the fake galaxies are configured by several input parameters and they are convolved with psf before being put on the image. The simulated sources are uniformly distributed in the CCD ( so are in all the provided configuration) with the number density of ``ngals_arcmin2``.

    .. seealso:: ``comest.ComEst.BulDiskLocator`` for more details about the configuration.

    """

    # set up the cpu number
    if  ncpu > multiprocessing.cpu_count():
        print
        print RuntimeWarning("ncpu:", ncpu, "is larger than total number of cpu:", multiprocessing.cpu_count())
        print RuntimeWarning("Using the total number of the cpu:", multiprocessing.cpu_count())
        print
        nproc    =   multiprocessing.cpu_count()
    else:
        nproc    =   ncpu

    # ---
    # Start simulate the image one by one
    # ---
    # set up the task queue
    task_queue          =   multiprocessing.Queue()
    # set up the done task
    done_queue          =   multiprocessing.Queue()

    # looping over images
    for nimage in xrange(nsimimages):
        # put the task
        task_queue.put(
            ( (path2image, zeropoint, psf_dict, stamp_size_arcsec, mag_dict, hlr_dict, fbulge_dict, q_dict, pos_ang_dict, ngals_arcmin2, random_seed + nimage),
            "simulated %i image" % nimage ) )

    # Run the tasks and create done_queue
    # Each Process command starts up a parallel process that will keep checking the queue
    # for a new task. If there is one there, it grabs it and does it. If not, it waits
    # until there is one to grab. When it finds a 'STOP', it shuts down.
    done_queue = multiprocessing.Queue()
    for nnk in xrange(nproc):
        multiprocessing.Process(target = _worker, args = ( _BulDiskLocator_single, task_queue, done_queue) ).start()

    # In the meanwhile, the main process keeps going.  We pull each image off of the
    # done_queue and put it in the appropriate place on the main image.
    # This loop is happening while the other processes are still working on their tasks.
    # You'll see that these logging statements get print out as the stamp images are still
    # being drawn.
    # claim the image
    image_collection    =   []
    # claim true_cats_collection
    true_cats_collection=   []
    # claim the processing_info
    info_list           =   []
    for nnk in xrange(nsimimages):
        single_image_and_cat, processing_info, processing_name      =       done_queue.get()
        print "#", "%s: for file %s was done." % (processing_name, processing_info)
        # info_list
        info_list.append( int( processing_info.strip("simulated image") ) )
        # append image
        image_collection.append(single_image_and_cat[0].copy())
        # append true catalog
        true_cats_collection.append(single_image_and_cat[1].copy())

    # Re-order according to the info_list - since the order of the image is important.
    image_collection    =   list(np.array(image_collection    )[ np.argsort(info_list) ])
    true_cats_collection=   list(np.array(true_cats_collection)[ np.argsort(info_list) ])

    # Stop the processes
    # The 'STOP's could have been put on the task list before starting the processes, or you
    # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
    # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
    # Once you are done with the processes, putting nproc 'STOP's will stop them all.
    # This is important, because the program will keep running as long as there are running
    # processes, even if the main process gets to the end.  So you do want to make sure to
    # add those 'STOP's at some point!
    for nnk in xrange(nproc):
        task_queue.put('STOP')

    # return
    return image_collection, true_cats_collection

def ModelGalLocator(
    path2image,
    readincat,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    nsimimages    = 10,
    random_seed   = 8241573,
    ncpu          = 2,
    ):
    """

    :param path2image: The absolute path to the image which you want to put the simulated sources on. This is is usually the source free image (SFI), or it can also be the BnB image if you want to simulate the sources on the image where the observed BnB sources are kept. One can uses BnB image to test how the BnB sources affect the detection.

    :param readincat: ndarray with the shape of (nsimimages, len(field names)), where field names are listed below. This array consists of the input catalog, including the input values with the field names of
    ``x_true``, ``y_true``, ``mag_true``, ``hlr_true[arcsec]``, ``fbulge_true``, ``q_true`` and ``pos_ang[deg]``.

    :param psf_dict: The psf configuration. Currently it only supports Moffat PSF with beta parameter of 4.5. ``psf_dict`` must be a dictionary in the form of ``{"moffat":{ "beta": _value_, "fwhm": _value_ } }``, where _value_ of ``fwhm`` is in the unit of arcsec. By default, ``psf_dict = {"moffat":{ "beta": 4.5, "fwhm": img_fwhm } }``.

    :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 20.0``.

    :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.

    :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.

    :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.

    :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.

    :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.

    :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 25.0``.

    :param nsimimages: The number of the images you want to simulate. It will be saved in the multi-extension file with the code name ``sims_nameroot``. By default, it is ``nsimimages = 50``.

    :param random_seed: The random seed of the random generator. It will be passed to **GalSim** for simulating the sources.

    :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**. By default, it is ``sims_nameroot = "buldisk"``.

    :param ncpu: The number of cpu for parallel running. By default, it is ``ncpu = 2``. Please do not set this number higher than the CPU cores you have.

    :type path2image: str
    :type readincat: ndarray
    :type psf_dict: dict
    :type stamp_size_arcsec: float
    :type mag_dict: dict
    :type hlr_dict: dict
    :type fbulge_dict: dict
    :type q_dict: dict
    :type pos_ang_dict: dict
    :type ngals_arcmin2: float
    :type nsimimages: int
    :type random_seed: int
    :type sims_nameroot: str
    :type ncpu: int

    :returns: ``image_collection`` is the list containing the simulated image and ``true_cats_collection`` is the list containing the information of the mock catalog (hence ``len(image_collection) = len(true_cats_collection) = nsimimages``). Each element of the lists is the simulated image outputed by Galsim.
    :rtype: list, list

    This function reads the ``path2image`` and put the fake galaxies on it. In this case it is the galaxies consisting of buldge and disk components. The properties of the fake galaxies are configured by several input parameters and they are convolved with psf before being put on the image. The simulated sources are uniformly distributed in the CCD ( so are in all the provided configuration) with the number density of ``ngals_arcmin2``.

    .. seealso:: ``comest.ComEst.BulDiskLocator`` for more details about the configuration.

    """

    # set up the cpu number
    if  ncpu > multiprocessing.cpu_count():
        print
        print RuntimeWarning("ncpu:", ncpu, "is larger than total number of cpu:", multiprocessing.cpu_count())
        print RuntimeWarning("Using the total number of the cpu:", multiprocessing.cpu_count())
        print
        nproc    =   multiprocessing.cpu_count()
    else:
        nproc    =   ncpu

    # ---
    # Start simulate the image one by one
    # ---
    # set up the task queue
    task_queue          =   multiprocessing.Queue()
    # set up the done task
    done_queue          =   multiprocessing.Queue()

    # get the nsimimages
    nsimimages          =   len(readincat)

    # looping over images
    for nimage in xrange(nsimimages):
        # put the task
        task_queue.put(
            ( (path2image, readincat[nimage], zeropoint, psf_dict, stamp_size_arcsec, mag_dict, hlr_dict, fbulge_dict, q_dict, pos_ang_dict, ngals_arcmin2, random_seed + nimage),
            "simulated %i image" % nimage ) )

    # Run the tasks and create done_queue
    # Each Process command starts up a parallel process that will keep checking the queue
    # for a new task. If there is one there, it grabs it and does it. If not, it waits
    # until there is one to grab. When it finds a 'STOP', it shuts down.
    done_queue = multiprocessing.Queue()
    for nnk in xrange(nproc):
        multiprocessing.Process(target = _worker, args = ( _ModelGalLocator_single, task_queue, done_queue) ).start()

    # In the meanwhile, the main process keeps going.  We pull each image off of the
    # done_queue and put it in the appropriate place on the main image.
    # This loop is happening while the other processes are still working on their tasks.
    # You'll see that these logging statements get print out as the stamp images are still
    # being drawn.
    # claim the image
    image_collection    =   []
    # claim true_cats_collection
    true_cats_collection=   []
    # claim the processing_info
    info_list           =   []
    for nnk in xrange(nsimimages):
        single_image_and_cat, processing_info, processing_name      =       done_queue.get()
        print "#", "%s: for file %s was done." % (processing_name, processing_info)
        # info_list
        info_list.append( int( processing_info.strip("simulated image") ) )
        # append image
        image_collection.append(single_image_and_cat.copy())
        #image_collection.append(single_image_and_cat[0].copy())
        # append true catalog
        #true_cats_collection.append(single_image_and_cat[1].copy())

    # Re-order according to the info_list - since the order of the image is important.
    image_collection    =   list(np.array(image_collection    )[ np.argsort(info_list) ])
    #true_cats_collection=   list(np.array(true_cats_collection)[ np.argsort(info_list) ])

    # Stop the processes
    # The 'STOP's could have been put on the task list before starting the processes, or you
    # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
    # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
    # Once you are done with the processes, putting nproc 'STOP's will stop them all.
    # This is important, because the program will keep running as long as there are running
    # processes, even if the main process gets to the end.  So you do want to make sure to
    # add those 'STOP's at some point!
    for nnk in xrange(nproc):
        task_queue.put('STOP')

    # return
    return image_collection



def RealGalLocator(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    nsimimages    = 10,
    random_seed   = 8241573,
    ncpu          = 2,
    ):
    """

    :param path2image: The absolute path to the image which you want to put the simulated sources on. This is is usually the source free image (SFI), or it can also be the BnB image if you want to simulate the sources on the image where the observed BnB sources are kept. One can uses BnB image to test how the BnB sources affect the detection.

    :param psf_dict: The psf configuration. Currently it only supports Moffat PSF with beta parameter of 4.5. ``psf_dict`` must be a dictionary in the form of ``{"moffat":{ "beta": _value_, "fwhm": _value_ } }``, where _value_ of ``fwhm`` is in the unit of arcsec. By default, ``psf_dict = {"moffat":{ "beta": 4.5, "fwhm": img_fwhm } }``.

    :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 20.0``.

    :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.

    :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.

    :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.

    :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.

    :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.

    :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 25.0``.

    :param nsimimages: The number of the images you want to simulate. It will be saved in the multi-extension file with the code name ``sims_nameroot``. By default, it is ``nsimimages = 50``.

    :param random_seed: The random seed of the random generator. It will be passed to **GalSim** for simulating the sources.

    :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**. By default, it is ``sims_nameroot = "buldisk"``.

    :param ncpu: The number of cpu for parallel running. By default, it is ``ncpu = 2``. Please do not set this number higher than the CPU cores you have.

    :type path2image: str
    :type psf_dict: dict
    :type stamp_size_arcsec: float
    :type mag_dict: dict
    :type hlr_dict: dict
    :type fbulge_dict: dict
    :type q_dict: dict
    :type pos_ang_dict: dict
    :type ngals_arcmin2: float
    :type nsimimages: int
    :type random_seed: int
    :type sims_nameroot: str
    :type ncpu: int

    :returns: ``image_collection`` is the list containing the simulated image and ``true_cats_collection`` is the list containing the information of the mock catalog (hence ``len(image_collection) = len(true_cats_collection) = nsimimages``). Each element of the lists is the simulated image outputed by Galsim.
    :rtype: list, list


    This function reads the ``path2image`` and put the fake galaxies on it. The properties of the fake galaxies are directly taken from **COSMOS** catalog provided by **Galsim** team and they are convolved with psf before being put on the image. In this routine, we only use ``pos_ang_dict`` to re-rotate the galaxies. Please run ``galsim_download_cosmos`` provided by **GalSim** to download the **COSMOS** catalog.  Please note that since we are resampling from the observed catalog, hence the configurations of the galaxy shape of ``hlr_dict``, ``fbulge_dict`` and ``q_dict`` do _NOT_ apply on this set of simulation. But for the sake of consistency,this routine still requires these input configuration. The simulated sources are uniformly distributed in the CCD with the number density of ``ngals_arcmin2``.

    .. seealso:: ``comest.ComEst.RealGalLocator`` for more details about the configuration.

    .. todo:: Perhaps we should add the input catalog arguments in the future.

    """

    # set up the cpu number
    if  ncpu > multiprocessing.cpu_count():
        print
        print RuntimeWarning("ncpu:", ncpu, "is larger than total number of cpu:", multiprocessing.cpu_count())
        print RuntimeWarning("Using the total number of the cpu:", multiprocessing.cpu_count())
        print
        nproc    =   multiprocessing.cpu_count()
    else:
        nproc    =   ncpu

    # ---
    # Start simulate the image one by one
    # ---
    # set up the task queue
    task_queue          =   multiprocessing.Queue()
    # set up the done task
    done_queue          =   multiprocessing.Queue()

    # looping over images
    for nimage in xrange(nsimimages):
        # put the task
        task_queue.put(
            ( (path2image, zeropoint, psf_dict, stamp_size_arcsec, mag_dict, hlr_dict, fbulge_dict, q_dict, pos_ang_dict, ngals_arcmin2, random_seed + nimage),
            "simulated %i image" % nimage ) )

    # Run the tasks and create done_queue
    # Each Process command starts up a parallel process that will keep checking the queue
    # for a new task. If there is one there, it grabs it and does it. If not, it waits
    # until there is one to grab. When it finds a 'STOP', it shuts down.
    done_queue = multiprocessing.Queue()
    for nnk in xrange(nproc):
        multiprocessing.Process(target = _worker, args = ( _RealGalLocator_single, task_queue, done_queue) ).start()

    # In the meanwhile, the main process keeps going.  We pull each image off of the
    # done_queue and put it in the appropriate place on the main image.
    # This loop is happening while the other processes are still working on their tasks.
    # You'll see that these logging statements get print out as the stamp images are still
    # being drawn.
    # claim the image
    image_collection    =   []
    # claim true_cats_collection
    true_cats_collection=   []
    # claim the processing_info
    info_list           =   []
    for nnk in xrange(nsimimages):
        single_image_and_cat, processing_info, processing_name      =       done_queue.get()
        print "#", "%s: for file %s was done." % (processing_name, processing_info)
        # info_list
        info_list.append( int( processing_info.strip("simulated image") ) )
        # append image
        image_collection.append(single_image_and_cat[0].copy())
        # append true catalog
        true_cats_collection.append(single_image_and_cat[1].copy())

    # Re-order according to the info_list - since the order of the image is important.
    image_collection    =   list(np.array(image_collection    )[ np.argsort(info_list) ])
    true_cats_collection=   list(np.array(true_cats_collection)[ np.argsort(info_list) ])

    # Stop the processes
    # The 'STOP's could have been put on the task list before starting the processes, or you
    # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
    # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
    # Once you are done with the processes, putting nproc 'STOP's will stop them all.
    # This is important, because the program will keep running as long as there are running
    # processes, even if the main process gets to the end.  So you do want to make sure to
    # add those 'STOP's at some point!
    for nnk in xrange(nproc):
        task_queue.put('STOP')

    # return
    return image_collection, true_cats_collection


def PntSrcLocator(
    path2image,
    zeropoint  = 20.789,
    psf_dict   = {"moffat":{ "beta": 4.5, "fwhm": 1.6} },
    stamp_size_arcsec   =   20.0,
    mag_dict   = {"lo":19.0, "hi":23.0 },
    hlr_dict   = {"lo":0.35 , "hi":0.75  },
    fbulge_dict= {"lo":0.5 , "hi":0.9  },
    q_dict     = {"lo":0.4 , "hi":1.0  },
    pos_ang_dict={"lo":0.0 , "hi":180.0},
    ngals_arcmin2 = 30.0,
    nsimimages    = 10,
    random_seed   = 8241573,
    ncpu          = 2,
    ):
    """

    :param path2image: The absolute path to the image which you want to put the simulated sources on. This is is usually the source free image (SFI), or it can also be the BnB image if you want to simulate the sources on the image where the observed BnB sources are kept. One can uses BnB image to test how the BnB sources affect the detection.

    :param psf_dict: The psf configuration. Currently it only supports Moffat PSF with beta parameter of 4.5. ``psf_dict`` must be a dictionary in the form of ``{"moffat":{ "beta": _value_, "fwhm": _value_ } }``, where _value_ of ``fwhm`` is in the unit of arcsec. By default, ``psf_dict = {"moffat":{ "beta": 4.5, "fwhm": img_fwhm } }``.

    :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 20.0``.

    :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.

    :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.

    :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.

    :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.

    :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.

    :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 25.0``.

    :param nsimimages: The number of the images you want to simulate. It will be saved in the multi-extension file with the code name ``sims_nameroot``. By default, it is ``nsimimages = 50``.

    :param random_seed: The random seed of the random generator. It will be passed to **GalSim** for simulating the sources.

    :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**. By default, it is ``sims_nameroot = "buldisk"``.

    :param ncpu: The number of cpu for parallel running. By default, it is ``ncpu = 2``. Please do not set this number higher than the CPU cores you have.

    :type path2image: str
    :type psf_dict: dict
    :type stamp_size_arcsec: float
    :type mag_dict: dict
    :type hlr_dict: dict
    :type fbulge_dict: dict
    :type q_dict: dict
    :type pos_ang_dict: dict
    :type ngals_arcmin2: float
    :type nsimimages: int
    :type random_seed: int
    :type sims_nameroot: str
    :type ncpu: int

    :returns: ``image_collection`` is the list containing the simulated image and ``true_cats_collection`` is the list containing the information of the mock catalog (hence ``len(image_collection) = len(true_cats_collection) = nsimimages``). Each element of the lists is the simulated image outputed by Galsim.
    :rtype: list, list


    This function reads the ``path2image`` and put the fake point sources on it. This method puts the fake sources on this image. In this case it is for the point sources (e.g., stars) convolving with the given PSF. Please note that since we are simulating point sources, the actually shape of simulated sources is just the PSF with the size of given ``img_fwhm``. Therefore the configurations of ``hlr_dict``, ``fbulge_dict``, ``q_dict`` and ``pos_ang_dict`` do _NOT_ apply here. But for the sake of consistency,this routine still requires these input configuration. The simulated sources are uniformly distributed in the CCD with the number density of ``ngals_arcmin2``.

    .. seealso:: ``comest.ComEst.PntSrcLocator`` for more details about the configuration.

    """

    # set up the cpu number
    if  ncpu > multiprocessing.cpu_count():
        print
        print RuntimeWarning("ncpu:", ncpu, "is larger than total number of cpu:", multiprocessing.cpu_count())
        print RuntimeWarning("Using the total number of the cpu:", multiprocessing.cpu_count())
        print
        nproc    =   multiprocessing.cpu_count()
    else:
        nproc    =   ncpu

    # ---
    # Start simulate the image one by one
    # ---
    # set up the task queue
    task_queue          =   multiprocessing.Queue()
    # set up the done task
    done_queue          =   multiprocessing.Queue()

    # looping over images
    for nimage in xrange(nsimimages):
        # put the task
        task_queue.put(
            ( (path2image, zeropoint, psf_dict, stamp_size_arcsec, mag_dict, hlr_dict, fbulge_dict, q_dict, pos_ang_dict, ngals_arcmin2, random_seed + nimage),
            "simulated %i image" % nimage ) )

    # Run the tasks and create done_queue
    # Each Process command starts up a parallel process that will keep checking the queue
    # for a new task. If there is one there, it grabs it and does it. If not, it waits
    # until there is one to grab. When it finds a 'STOP', it shuts down.
    done_queue = multiprocessing.Queue()
    for nnk in xrange(nproc):
        multiprocessing.Process(target = _worker, args = ( _PntSrcLocator_single, task_queue, done_queue) ).start()

    # In the meanwhile, the main process keeps going.  We pull each image off of the
    # done_queue and put it in the appropriate place on the main image.
    # This loop is happening while the other processes are still working on their tasks.
    # You'll see that these logging statements get print out as the stamp images are still
    # being drawn.
    # claim the image
    image_collection    =   []
    # claim true_cats_collection
    true_cats_collection=   []
    # claim the processing_info
    info_list           =   []
    for nnk in xrange(nsimimages):
        single_image_and_cat, processing_info, processing_name      =       done_queue.get()
        print "#", "%s: for file %s was done." % (processing_name, processing_info)
        # info_list
        info_list.append( int( processing_info.strip("simulated image") ) )
        # append image
        image_collection.append(single_image_and_cat[0].copy())
        # append true catalog
        true_cats_collection.append(single_image_and_cat[1].copy())

    # Re-order according to the info_list - since the order of the image is important.
    image_collection    =   list(np.array(image_collection    )[ np.argsort(info_list) ])
    true_cats_collection=   list(np.array(true_cats_collection)[ np.argsort(info_list) ])

    # Stop the processes
    # The 'STOP's could have been put on the task list before starting the processes, or you
    # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
    # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
    # Once you are done with the processes, putting nproc 'STOP's will stop them all.
    # This is important, because the program will keep running as long as there are running
    # processes, even if the main process gets to the end.  So you do want to make sure to
    # add those 'STOP's at some point!
    for nnk in xrange(nproc):
        task_queue.put('STOP')

    # return
    return image_collection, true_cats_collection

