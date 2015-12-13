#!/usr/bin/env python

######
#
# Module for Completeness Estimators
#
######

import numpy as np
from math import *
#import scipy.interpolate as interpolate
#import matplotlib.pyplot as pyplt
#import galsim
import pyfits
import SrcPlacer
import py2dmatch
import numpy.lib.recfunctions as rfn # magic, used to merge structured array.
from distutils.spawn import find_executable
import os
import time
import sys

PATH2CODE        =       os.path.dirname( os.path.abspath(__file__) ) # get the current directory

######
#
# Define functions
#
######

def SE_CMD_CREATOR(se_exec_is, img_is, se_config_is, se_params_is, outdir_is, name_root_is, args_is = "", outputcheckimage = True, next = 0):
    """
    This method will create the se command which is run on the image.
        
    Parameters:
        -`se_exec_is`: string. The path to to bin of SE. Usually it is the output of `which sex`.
        -`img_is`: string. The abs path of the image which we wish to run SE.
        -`se_config_is`: string. The abs path of SE config.
        -`se_params_is`: string. The abs path of SE params.
        -`outdir_is`: string. The abs directory that we wish to put SE output.
        -`name_root_is`: string. The root name of the output of SE. E.g., [name_root_is].cat.fits for catalog.
        -`args_is`: string. The additional arguments of SE. sex img_is -c se_config_is args_is
        -`outputcheckimage`: bool. Whether or not to output the check image.
        -`next`: non-negative int. The next you want to run sex on. Default is zero.
    Returns:
        -`cmd_is`: string. The command which can be called via os.system(cmd_is).
                   This is the command used to run SE.
    """
    # out products defining
    catalog_name                    =       os.path.join(outdir_is, name_root_is + ".cat.fits")
    check_identical                 =       os.path.join(outdir_is, name_root_is + ".identical.fits")
    check_background                =       os.path.join(outdir_is, name_root_is + ".background.fits")
    check_background_rms            =       os.path.join(outdir_is, name_root_is + ".background_rms.fits")
    check_minus_background          =       os.path.join(outdir_is, name_root_is + ".-background.fits")
    check_objects                   =       os.path.join(outdir_is, name_root_is + ".objects.fits")
    check_minus_objects             =       os.path.join(outdir_is, name_root_is + ".-objects.fits")
    check_segmentation              =       os.path.join(outdir_is, name_root_is + ".segmentation.fits")
    check_apertures                 =       os.path.join(outdir_is, name_root_is + ".apertures.fits")
    # construct the CHECKIMAGE_NAME
    checkimage_name                 =       check_identical        +    "," + \
                                            check_background       +    "," + \
                                            check_background_rms   +    "," + \
                                            check_minus_background +    "," + \
                                            check_objects          +    "," + \
                                            check_minus_objects    +    "," + \
                                            check_segmentation     +    "," + \
                                            check_apertures
    # construct the CHECKIMAGE_TYPE
    checkimage_type                 =       "IDENTICAL,BACKGROUND,BACKGROUND_RMS,-BACKGROUND,OBJECTS,-OBJECTS,SEGMENTATION,APERTURES"
    # construct the SE command - default is that we use filter
    if outputcheckimage == True:
        cmd_is  =                     se_exec_is      + "    " + \
                                      img_is          + "["+str(next)+"]     " + \
              " -c "                + se_config_is    + "    " + \
              " -PARAMETERS_NAME "  + se_params_is    + "    " + \
              " -CHECKIMAGE_TYPE "  + checkimage_type + "    " + \
              " -CHECKIMAGE_NAME "  + checkimage_name + "    " + \
              " -CATALOG_NAME    "  + catalog_name    + "    " + \
              args_is
    else:
        cmd_is  =                     se_exec_is      + "    " + \
                                      img_is          + "["+str(next)+"]     " + \
              " -c "                + se_config_is    + "    " + \
              " -PARAMETERS_NAME "  + se_params_is    + "    " + \
              " -CHECKIMAGE_TYPE "  + "NONE"          + "    " + \
              " -CHECKIMAGE_NAME "  + checkimage_name + "    " + \
              " -CATALOG_NAME    "  + catalog_name    + "    " + \
              args_is

    return cmd_is


def CreateSrcFreeMap(idnt_map, segm_map, bckg_map, bckg_rms_map, objc_map, path2out_map):
    """
    This function reads in the check image of SE outputs running on the SAME input image and eventually 
    creates the full source-free fits image.
    
    It will copy the identical map and filter the pixels for the given segmentation map.
    If the pixel is not belonging to a source, then nothing happenes.
    If the pixel is belonging to a source, then this function will generate the background value for that pixel
    given the same position at the background and background_rms check images outputted by SE.
    
    Parameters:
        -`idnt_map` :   string. the path to the identical fits image.
        -`segm_map` :   string. the path to the segmentation fits image.
        -`bckg_map` :   string. the path to the background fits image.
        -`bckg_rms_map` :   string. the path to the background_rms fits image.
        -`objc_map` :   string. the path to the objects fits image.
        -`path2out_map` :   string. the path to the identical fits image.
        
    Returns:
        This functions will save the source-free fits image in the file named path2out_map.
    """
    # read in fits files and save them into the dict
    readin_imgs     =   {
        "idnt_map"      :   pyfits.getdata(idnt_map),
        "segm_map"      :   pyfits.getdata(segm_map),
        "bckg_map"      :   pyfits.getdata(bckg_map),
        "bckg_rms_map"  :   pyfits.getdata(bckg_rms_map),
        "objc_map"      :   pyfits.getdata(objc_map),
    }
    readin_headers  =   {
        "idnt_map"      :   pyfits.getheader(idnt_map, ext = -1),
        "segm_map"      :   pyfits.getheader(segm_map, ext = -1),
        "bckg_map"      :   pyfits.getheader(bckg_map, ext = -1),
        "bckg_rms_map"  :   pyfits.getheader(bckg_rms_map, ext = -1),
        "objc_map"      :   pyfits.getheader(objc_map, ext = -1),
    }

    # filter the i_am_objs_pixels and i_am_bckg_pixels
    i_am_bckg_pixels    =   ( readin_imgs["segm_map"] == 0 )
    i_am_objs_pixels    =   ~i_am_bckg_pixels

    # create the map.
    out_map             =   np.copy( readin_imgs["idnt_map"] )
    
    # simulate the background
    try:
        simulated_bckg      =   np.random.normal(
                                loc   = readin_imgs["bckg_map"    ],
                                scale = readin_imgs["bckg_rms_map"] )
    except ValueError:
        print RuntimeWarning("background rms map from Sextractor has value < 0, probably due to data corruption. We use median value instead.")
        bckg_rms_map_sanitized        = np.copy( readin_imgs["bckg_rms_map"] )
        keep_me_in_rms_map            = np.isfinite( readin_imgs["bckg_rms_map"] ) & ( readin_imgs["bckg_rms_map"] > 0.0 )
        bckg_rms_map_sanitized[ ~keep_me_in_rms_map ]   = np.median( bckg_rms_map_sanitized[ keep_me_in_rms_map ] )
        simulated_bckg      =   np.random.normal(
                                loc   = readin_imgs["bckg_map"    ],
                                scale = bckg_rms_map_sanitized )
                                #scale = np.abs(readin_imgs["bckg_rms_map"]) )
        # clean
        del bckg_rms_map_sanitized, keep_me_in_rms_map

    # fill the object pixels with the simulated background in out_map
    out_map[i_am_objs_pixels]     =     simulated_bckg[i_am_objs_pixels]
                                             
    # write the outfits
    out_hdu             =   pyfits.PrimaryHDU(
                            data = out_map, header = readin_headers["idnt_map"] )
    out_hdu.writeto(path2out_map, clobber = True)
                                             
    return 0

def CreateBnBMap(idnt_map, segm_map, bckg_map, bckg_rms_map, objc_map, srcfree, path2out_map):
    """
    This function reads in the check image of SE outputs running on the SAME input image and eventually 
    creates the full fits image with only BnB (Big and Bright) objects.
    
    It will copy the identical map and filter the pixels for the given segmentation map.
    If the pixel is belonging to a source, then nothing happenes.
    If the pixel is not belonging to a source, then this function will generate the background value for that pixel
    given the same position at the background and background_rms check images outputted by SE.
    
    Caveat: it has to use the source-free image created previously by CreateSrcFreeMap.
    
    Parameters:
        -`idnt_map` :   string. the path to the identical fits image.
        -`segm_map` :   string. the path to the segmentation fits image.
        -`bckg_map` :   string. the path to the background fits image.
        -`bckg_rms_map` :   string. the path to the background_rms fits image.
        -`objc_map` :   string. the path to the objects fits image.
        -`srcfree`  :   string. the path to the source free image created previously.
                        It has to be exactly the same size and dimensions as the idnt_map.
        -`path2out_map` :   string. the path to the identical fits image.
        
    Returns:
        This functions will save the source-free fits image in the file named path2out_map.
    """
    # read in fits files and save them into the dict
    readin_imgs     =   {
        "idnt_map"      :   pyfits.getdata(idnt_map),
        "segm_map"      :   pyfits.getdata(segm_map),
        "bckg_map"      :   pyfits.getdata(bckg_map),
        "bckg_rms_map"  :   pyfits.getdata(bckg_rms_map),
        "objc_map"      :   pyfits.getdata(objc_map),
        "srcfree"       :   pyfits.getdata(srcfree),
    }
    readin_headers  =   {
        "idnt_map"      :   pyfits.getheader(idnt_map, ext = -1),
        "segm_map"      :   pyfits.getheader(segm_map, ext = -1),
        "bckg_map"      :   pyfits.getheader(bckg_map, ext = -1),
        "bckg_rms_map"  :   pyfits.getheader(bckg_rms_map, ext = -1),
        "objc_map"      :   pyfits.getheader(objc_map, ext = -1),
        "srcfree"       :   pyfits.getheader(srcfree, ext = -1),
    }

    # filter the i_am_objs_pixels and i_am_bckg_pixels
    i_am_bckg_pixels    =   ( readin_imgs["segm_map"] == 0 )
    i_am_objs_pixels    =   ~i_am_bckg_pixels

    # create the map.
    out_map             =   np.copy( readin_imgs["idnt_map"] )
        
    # fill the object pixels with the simulated background in out_map
    out_map[i_am_bckg_pixels]     =     np.copy( readin_imgs["srcfree"] )[i_am_bckg_pixels]
                                             
    # write the outfits
    out_hdu             =   pyfits.PrimaryHDU(
                            data = out_map, header = readin_headers["idnt_map"] )
    out_hdu.writeto(path2out_map, clobber = True)
                                             
    return 0

def Nearest_interpolator(values_array, x_edges, y_edges, new_x_edges, new_y_edges):
    """
    This subroutine interpolate the values of given arrays by 'Nearest_Neighbor' algorithm.
    
    This function uses scipy.interpolate.NearestNDInterpolator
    
    Parameters:
        -`values_array`: 2d np array. The map of the values in the cooridinate of [ny, nx], 
                         where nx and ny are the dimensions of the x and y coordinate, respectively.
        -`x_edges`: 1d array. The edges of binning of x-coordinate. It has lengh of nx+1.
        -`y_edges`: 1d array. The edges of binning of y-coordinate. It has lengh of ny+1.
        -`new_x_edges`: 1d array. The new edges of the binning of x-coordinate.
        -`new_y_edges`: 1d array. The new edges of the binning of y-coordinate.
    Return:
        -`new_values_array`: 2d np array. The new map with the value interpolated by 'Nearest Neighbor' Algorithm at the 
                             coordinates assigned in the binning with the edges of new_x[y]_edges. 
                             It has the shape of [my, mx], wehre my + 1 = len(new_x_edges) and mx + 1 = len(new_y_edges).
    """
    # derive x_bins and y_bins
    x_bins  =   0.5*( x_edges[1:] + x_edges[:-1] )
    y_bins  =   0.5*( y_edges[1:] + y_edges[:-1] )
    # derive x_mesh and y_mesh
    x_mesh, y_mesh = np.meshgrid(x_bins, y_bins)
    # for new_x_bins and new_y_bins
    new_x_bins  =   0.5*( new_x_edges[1:] + new_x_edges[:-1] )
    new_y_bins  =   0.5*( new_y_edges[1:] + new_y_edges[:-1] )
    # derive x_mesh and y_mesh
    new_x_mesh, new_y_mesh = np.meshgrid(new_x_bins, new_y_bins)
    # derive the interpolator
    interp_seed    = interpolate.NearestNDInterpolator( 
                     np.vstack( (x_mesh.flatten(), y_mesh.flatten()) ).T, 
                     values_array.flatten() )
    # derive the new_values_array
    new_values_array    =   interp_seed(new_x_mesh.flatten(), new_y_mesh.flatten()).reshape(new_x_mesh.shape)
    return new_values_array

######
#
# module
#
######

class fitsimage:

    # ---
    # initialize the set ups
    # ---
    def __init__(self, path2img,
                       path2outdir,
                       img_zp,
                       img_pixel_scale,
                       img_fwhm,
                       sex_exec          =   find_executable("sex"),
                       sex_config        =   os.path.join(PATH2CODE, "templates/sex.config"),
                       sex_params        =   os.path.join(PATH2CODE, "templates/sex.params"),
                       full_root_name    =   "full",
                       bnb_root_name     =   "bnb",
                       full_sex_args     =   "-FILTER_NAME     "  + os.path.join(PATH2CODE, "templates/filters/default.conv"),
                       bnb_sex_args      =   "-FILTER_NAME     "  + os.path.join(PATH2CODE, "templates/filters/default.conv") + "    " + "-DETECT_MINAREA 10 -DETECT_THRESH 5 -ANALYSIS_THRESH 5",   
                 ):
        """
        
        :param path2img: The absolute path to the observed image. It is the the only input file required by **ComEst**. It is _neccessary_ to have the header with correct WCS (``CD?_?``, ``CRVAL?`` and ``CRPIX?``) in the image of ``path2img``.
        
        :param path2outdir: The absolute path to the output directory. All the files created by **ComEst** are saved in this directory. **ComEst** will create ``path2outdir`` if ``path2outdir`` does not exist (therefore please make sure you have the permission to create such directory).
        
        :param img_zp: The zeropoint of the observed image. **ComEst** estimates the analysis based on the the given zeropoint. Specifically, it is the zeropoint that will be passed to **SExtractor** for extracting the photometry. Please make sure you have zeropoint well-calibrated. Currently **ComEst** does not support reading ZP from the header.
        
        :param img_pixel_scale: The pixel scale of the observed image. It is in the unit of arcsec/pixel. Currently **ComEst** does not support reading pixel scale from the header while loading in the image.
        
        :param img_fwhm: The Full Width Half Maximum of the observed image. The sources simulated by **ComEst** are convolved with the Point Spread Function (PSF) with the FWHM of ``img_fwhm``.
        
        :param sex_exec: The absolute path to the executable **SExtactor**. **ComEst** will look for **SExtractor** in ``$PATH``, by Design. You can of course directly assign the absolute path of **SExtractor** if you desire. Once ``sex_exec`` is desided for this instance, then it will be the absolute path to **SExtractor** afterward.
        
        :param sex_config: The configuration of **SExtractor**. **ComEst** will load the configuration ``sex.config`` in the directory ``templates`` of the source code. You can of course directly assign the absolute path of **SExtractor** config in this stage.
        
        :param sex_params: The output parameter file of **SExtractor**. **ComEst** will load the configuration ``sex.params`` in the directory ``templates`` of the source code. You can of course directly assign the absolute path of **SExtractor** parameter file in this stage. **IMPORTANT**: the parameters file MUST contain the output parameters listed in parameter file in the directory of ``templates``. The best recommendation is just to modify the parameter file in the directory of ``templates``.
        
        :param full_root_name: The code name for the products of **SExtractor** on the first run, which is designed to detect all the observed sources. Default is "full".
        
        :param bnb_root_name: The code name for the products of **SExtractor** on the run, which is designed to detect all the Big and Bright (BnB) sources. This is done by tuning the **SExtractor** configuration. Default is "bnb". This run is useful when one want to create a rough mask map of the image or to investigate how the masking effect from the BnB sources affects the analysis.
        
        :param full_sex_args: The arguments of **SExtractor** for the "full" run. This argument is exactly the argument which will be passed to **SExtractor**. You _have_ to specifically assign the ``FILTER_NAME`` since **SExtractor** does not support automatically loading the filter. For example, you can assign ``full_sex_args = "-FILTER_NAME /path/to/the/filter/file -CATALOG_NAME /path/to/output/cat"`` to make **SExtractor** have the customized ``FILTER_NAME`` and ``CATALOG_NAME`` in the configuration file. If one wants to use weight image in **SExtractor** run, the configuration for weighting should be included in this argument. See ``sex_config`` for the default configuration of **SExtractor**. **ComEst** uses ``"-FILTER_NAME /path/to/source/code/templates/filters/default.conv"`` by default.
        
        :param bnb_sex_args: The arguments of **SExtractor** for the "bnb" run. This argument is exactly the argument which will be passed to **SExtractor**. You _have_ to specifically assign the ``FILTER_NAME`` since **SExtractor** does not support automatically loading the filter. For example, you can assign ``full_sex_args = "-FILTER_NAME /path/to/the/filter/file -CATALOG_NAME /path/to/output/cat"`` to make **SExtractor** have the customized ``FILTER_NAME`` and ``CATALOG_NAME`` in the configuration file. If one wants to use weight image in **SExtractor** run, the configuration for weighting should be included in this argument. See ``sex_config`` for the default configuration of **SExtractor**. **ComEst** uses ``"-FILTER_NAME /path/to/source/code/templates/filters/default.conv -DETECT_MINAREA 10 -DETECT_THRESH 5 -ANALYSIS_THRESH 5"`` by default. This means **ComEst** will identify the sources with ``DETECT_THRESH>5`` and ``DETECT_MINAREA>10pixel`` as the BnB sources.
        
        :type path2img: str
        :type path2outdir: str
        :type img_zp: float
        :type img_pixel_scale: float
        :type img_fwhm: float
        :type sex_exec: str
        :type sex_config: str
        :type sex_params: str
        :type full_root_name: str
        :type bnb_root_name: str
        :type full_sex_args: str
        :type bnb_sex_args: str
        
        Initialize the observed image saved in the FITS format.
        """
        self.path2img           =   path2img
        self.path2outdir        =   path2outdir
        self.sex_exec           =   sex_exec
        self.sex_config         =   sex_config
        self.sex_params         =   sex_params
        self.full_root_name     =   full_root_name
        self.bnb_root_name      =   bnb_root_name
        self.full_sex_args      =   full_sex_args
        self.bnb_sex_args       =   bnb_sex_args
        self.img_zp             =   float(img_zp)
        self.img_pixel_scale    =   float(img_pixel_scale)
        self.img_fwhm           =   float(img_fwhm)
    
        # set up the number of pixels of image
        self.y_npixels, self.x_npixels  =   pyfits.getdata(path2img).shape

    # ---
    # diagnostic output
    # ---
    def i_am(self):
        """
        print the diagnostic information
        """
        print
        print "#", "fits image information as below:"
        print "#", "path2img:", self.path2img
        print "#", "path2outdir:", self.path2outdir
        print "#", "sex_exec:", self.sex_exec
        print "#", "sex_full_config:", self.sex_config
        print "#", "sex_full_params:", self.sex_params
        print "#", "full_root_name:", self.full_root_name
        print "#", "bnb_root_name:", self.bnb_root_name
        print "#", "full_sex_args:", self.full_sex_args
        print "#", "bnb_sex_args:", self.bnb_sex_args
        print "#", "img_zp:", self.img_zp
        print "#", "img_pixel_scale:", self.img_pixel_scale, "[arcsec/pix]"
        print "#", "img_fwhm:", self.img_fwhm, "[arcsec]"
        print "#", "x_npixels:", self.x_npixels, "[pix]"
        print "#", "y_npixels:", self.y_npixels, "[pix]"
        print

    # ---
    # Run SE command
    # ---
    def RunSE(self, name_root_is, se_args_is):
        """
        
        :param name_root_is: The code name for this **SExtractor** run. The names of all outputs (catalog/check-images) of **SExtractor** are marked by this code name consistently.
        
        :param se_args_is: The arguments of **SExtractor**. This argument is exactly the argument which will be passed to **SExtractor**. For example, you can assign ``se_args_is = "-FILTER_NAME /path/to/the/filter/file -CATALOG_NAME /path/to/output/cat"`` to make **SExtractor** have the customized ``FILTER_NAME`` and ``CATALOG_NAME`` in the configuration file.
        
        :type name_root_is: str
        :type se_args_is: str
        
        :returns: ``stdout``. The standard output status of **SExtractor** run. ``stdout = 0`` for successful **SExtractor** run. The output files are saved in the directory ``path2outdir``.
        :rtype: int
        
        This method will create the se command which will be run on the observed image. The output files have consistent code name of ``name_root_is``.
        """
        # construct the args
        put_me_in_se_args =   " -MAG_ZEROPOINT" + "    " + "%.3f" % self.img_zp   + \
                              " -SEEING_FWHM"   + "    " + "%.3f" % self.img_fwhm + \
                              " -PIXEL_SCALE"   + "    " + "%.3f" % self.img_pixel_scale + \
                              " -FILTER_NAME"   + "    " + os.path.join(PATH2CODE, "templates/default.conv") + \
                              "    " + se_args_is
        # construct the command
        run_me  =   SE_CMD_CREATOR(se_exec_is   = self.sex_exec,
                                   img_is       = self.path2img,
                                   se_config_is = self.sex_config,
                                   se_params_is = self.sex_params,
                                   outdir_is    = self.path2outdir,
                                   name_root_is = name_root_is,
                                   args_is      = put_me_in_se_args)
        # print run_me
        print "#", "run_me:"
        print
        print run_me
        print
        # make sure outdir_is exists
        if os.path.isdir(self.path2outdir) == False: os.makedirs(self.path2outdir)
        # run it!
        stdout      =       os.system(run_me)
        return stdout

    # ---
    # Run SE command for all sources
    # ---
    def RunSEforAll(self):
        """
        
        :returns: ``stdout``. The standard output status of **SExtractor** run. ``stdout = 0`` for successful **SExtractor** run.
        :rtype: int
        
        This method will create the se command which will be run on the observed image. The output files have consistent code name of ``full_root_name``. This method is design to catalog all the sources observed on the ``path2img``. The configuration (``sex_config``) and parameter files (``sex_params``) are used, while the argument ``full_sex_args`` is passed to **SExtractor**. The output files are saved in the directory ``path2outdir``.
        
        """
        # construct the args
        put_me_in_se_args =   " -MAG_ZEROPOINT" + "    " + "%.3f" % self.img_zp   + \
                              " -SEEING_FWHM"   + "    " + "%.3f" % self.img_fwhm + \
                              " -PIXEL_SCALE"   + "    " + "%.3f" % self.img_pixel_scale + \
                              "    " + self.full_sex_args
        # construct the command
        run_me  =   SE_CMD_CREATOR(se_exec_is   = self.sex_exec,
                                   img_is       = self.path2img,
                                   se_config_is = self.sex_config,
                                   se_params_is = self.sex_params,
                                   outdir_is    = self.path2outdir,
                                   name_root_is = self.full_root_name,
                                   args_is      = put_me_in_se_args)
        # print run_me
        print "#", "run_me:"
        print
        print run_me
        print
        # make sure outdir_is exists
        if os.path.isdir(self.path2outdir) == False: os.makedirs(self.path2outdir)
        # run it!
        stdout      =       os.system(run_me)
        return stdout

    # ---
    # Run SE command for BNB
    # ---
    def RunSEforBnB(self):
        """
            
        :returns: ``stdout``. The standard output status of **SExtractor** run. ``stdout = 0`` for successful **SExtractor** run.
        :rtype: int
            
        This method will create the se command which will be run on the observed image. The output files have consistent code name of ``bnb_root_name``. This method is design to catalog all the BnB (Bright and Big) sources observed on the ``path2img``. The configuration (``sex_config``) and parameter files (``sex_params``) are used, while the argument ``bnb_sex_args`` is passed to **SExtractor**. The output files are saved in the directory ``path2outdir``
        
        """
        # construct the args
        put_me_in_se_args =   " -MAG_ZEROPOINT" + "    " + "%.3f" % self.img_zp   + \
                              " -SEEING_FWHM"   + "    " + "%.3f" % self.img_fwhm + \
                              " -PIXEL_SCALE"   + "    " + "%.3f" % self.img_pixel_scale + \
                              "    " + self.bnb_sex_args
        # construct the command
        run_me  =   SE_CMD_CREATOR(se_exec_is   = self.sex_exec,
                                   img_is       = self.path2img,
                                   se_config_is = self.sex_config,
                                   se_params_is = self.sex_params,
                                   outdir_is    = self.path2outdir,
                                   name_root_is = self.bnb_root_name,
                                   args_is      = put_me_in_se_args)
        # print run_me
        print "#", "run_me:"
        print
        print run_me
        print
        # make sure outdir_is exists
        if os.path.isdir(self.path2outdir) == False: os.makedirs(self.path2outdir)
        # run it!
        stdout      =       os.system(run_me)
        return stdout

    # ---
    # Create src_free images
    # ---
    def SaveSrcFreeFits(self):
        """
            
        This function run CreateFullSrcFreeMap to create the source-free image (SFI). The output files are saved in the path2outdir with the name of ``full_root_name`` + ``.identical.srcfree.fits``.
        
        """
        # run the command
        CreateSrcFreeMap(
        idnt_map        =   os.path.join(self.path2outdir, self.full_root_name + ".identical.fits"),
        segm_map        =   os.path.join(self.path2outdir, self.full_root_name + ".segmentation.fits"),
        bckg_map        =   os.path.join(self.path2outdir, self.full_root_name + ".background.fits"),
        bckg_rms_map    =   os.path.join(self.path2outdir, self.full_root_name + ".background_rms.fits"),
        objc_map        =   os.path.join(self.path2outdir, self.full_root_name + ".objects.fits"),
        path2out_map    =   os.path.join(self.path2outdir, self.full_root_name + ".identical.srcfree.fits"),
        )
        # diagnostic
        print
        print "#", "Source-free image:",
        print os.path.join(self.path2outdir, self.full_root_name + ".identical.srcfree.fits")
        print
    
    # ---
    # Create bnb images
    # ---
    def SaveBnBFits(self):
        """
            
        This function run CreateBnBMap to create the BnB image. The output files are saved in the ``path2outdir`` with the name of ``bnb_root_name`` + ``.identical.bnb.fits``.
        
        """
        # run the command
        CreateBnBMap(
        idnt_map        =   os.path.join(self.path2outdir, self.bnb_root_name + ".identical.fits"),
        segm_map        =   os.path.join(self.path2outdir, self.bnb_root_name + ".segmentation.fits"),
        bckg_map        =   os.path.join(self.path2outdir, self.bnb_root_name + ".background.fits"),
        bckg_rms_map    =   os.path.join(self.path2outdir, self.bnb_root_name + ".background_rms.fits"),
        objc_map        =   os.path.join(self.path2outdir, self.bnb_root_name + ".objects.fits"),
        srcfree         =   os.path.join(self.path2outdir, self.full_root_name+ ".identical.srcfree.fits"),
        path2out_map    =   os.path.join(self.path2outdir, self.bnb_root_name + ".identical.bnb.fits"),
        )
        # diagnostic
        print
        print "#", "BnB image:",
        print os.path.join(self.path2outdir, self.bnb_root_name + ".identical.bnb.fits")
        print


    # ---
    # BulDiskLocator
    # ---
    def BulDiskLocator(self,
        path2image,
        psf_dict   = None,
        stamp_size_arcsec   =   20.0,
        mag_dict   = {"lo":20.0, "hi":25.0 },
        hlr_dict   = {"lo":0.35 , "hi":0.75  },
        fbulge_dict= {"lo":0.5 , "hi":0.9  },
        q_dict     = {"lo":0.4 , "hi":1.0  },
        pos_ang_dict={"lo":0.0 , "hi":180.0},
        ngals_arcmin2 = 15.0,
        nsimimages    = 50,
        random_seed   = 234231,
        sims_nameroot = "buldisk",
        ncpu          = 2,
        ):
        """
            
        :param path2image: The absolute path to the image which you want to put the simulated sources on. This is usually the source free image (SFI), or it can also be the BnB image if you want to simulate the sources on the image where the observed BnB sources are kept. One can uses BnB image to test how the BnB sources affect the detection.
        
        :param psf_dict: The psf configuration. Currently it only supports Moffat PSF with beta parameter of 4.5. ``psf_dict`` must be a dictionary in the form of ``{"moffat":{ "beta": _value_, "fwhm": _value_ } }``, where _value_ of ``fwhm`` is in the unit of arcsec. By default, ``psf_dict = {"moffat":{ "beta": 4.5, "fwhm": img_fwhm } }``.
        
        :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 15.0``.
        
        :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.
        
        :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.
        
        :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.
        
        :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.
        
        :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.
        
        :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 15.0``.
        
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
        
        :returns: ``out_mef`` is the list containing the simulated images and ``out_true_cats`` is the list containing the information of the mock catalogs (hence ``len(out_mef) = len(out_true_cats) = nsimimages``).
        :rtype: list, list
        
        This method calls the routine ``comest.SrcPlacer.BulDiskLocator`` to put the fake sources on this image. In this case it is the galaxies consisting of buldge and disk components. The simulated sources are uniformly distributed in the CCD ( so are in all the provided configuration) with the number density of ``ngals_arcmin2``.
        
        .. seealso:: ``comest.SrcPlacer.BulDiskLocator`` for more details about the configuration.
        
            
        """
        # assign the value and sanitize
        zeropoint   =   self.img_zp     # zeropoint
        # assign psf
        if  psf_dict  is None:
            psf_dict    =   {"moffat":{ "beta": 4.5, "fwhm": self.img_fwhm } }
            
        # diagnostic output
        print
        print "#", "Using BulDiskLocator to put galaxies (bulge + disk) on the targeted image..."
        print "#", "path2image:", path2image
        print "#", "psf_dict:", psf_dict
        print "#", "stamp_size_arcsec:", stamp_size_arcsec
        print "#", "mag_dict:", mag_dict
        print "#", "hlr_dict:", hlr_dict
        print "#", "fbulge_dict:", fbulge_dict
        print "#", "q_dict:", q_dict
        print "#", "pos_ang_dict:", pos_ang_dict
        print "#", "ngals_arcmin2:", ngals_arcmin2
        print "#", "nsimimages:", nsimimages
        print "#", "random_seed:", random_seed
        print "#", "sims_nameroot:", sims_nameroot
        print "#", "ncpu:", ncpu
        print
        
        # timing
        t1      =       time.time()
        
        # run it and save it as the file.
        out_mef, out_true_cats  = SrcPlacer.BulDiskLocator(
            path2image          = path2image,
            zeropoint           = zeropoint,
            psf_dict            = psf_dict,
            stamp_size_arcsec   = stamp_size_arcsec,
            mag_dict            = mag_dict,
            hlr_dict            = hlr_dict,
            fbulge_dict         = fbulge_dict,
            q_dict              = q_dict,
            pos_ang_dict        = pos_ang_dict,
            ngals_arcmin2       = ngals_arcmin2,
            nsimimages          = nsimimages,
            random_seed         = random_seed,
            ncpu                = ncpu,
            )
        
        # timing
        t2      =       time.time()
        
        # diagnostic
        print
        print "#", "Total time takes:", t2 - t1
        print
        
        # make sure dir exists
        outdir_sims             = os.path.join(self.path2outdir, sims_nameroot + "_sims")
        if os.path.isdir( outdir_sims ) == False: os.makedirs(outdir_sims)
        # save
        # for mef image
        SrcPlacer.galsim.fits.writeMulti(out_mef, outdir_sims + "/" + sims_nameroot + ".sims.fits")
        # for mef table
        phdulist = pyfits.PrimaryHDU( header = pyfits.BinTableHDU(data = out_true_cats[0]).header ) # primary table
        list4fits= [phdulist]                                                                       # put it in the list first
        for ntable in  xrange(nsimimages):                                                          # append all the tables
            list4fits.append( pyfits.BinTableHDU(data = out_true_cats[ntable]) )                    # append all the tables
        thdulist = pyfits.HDUList(list4fits)                                                        # convert it into table hdulist
        thdulist.writeto(outdir_sims + "/" + sims_nameroot + ".sims.cat.fits", clobber = True)      # save
        
        # diagnostic
        print
        print "#", "sims image:",
        print outdir_sims + "/" + sims_nameroot + ".sims.fits"
        print "#", "sims cat:",
        print outdir_sims + "/" + sims_nameroot + ".sims.cat.fits"
        print
        
        return out_mef, out_true_cats

    # ---
    # RealGalLocator
    # ---
    def RealGalLocator(self,
        path2image,
        psf_dict   = None,
        stamp_size_arcsec   =   20.0,
        mag_dict   = {"lo":20.0, "hi":25.0 },
        hlr_dict   = {"lo":0.35 , "hi":0.75  },
        fbulge_dict= {"lo":0.5 , "hi":0.9  },
        q_dict     = {"lo":0.4 , "hi":1.0  },
        pos_ang_dict={"lo":0.0 , "hi":180.0},
        ngals_arcmin2 = 15.0,
        nsimimages    = 50,
        random_seed   = 234231,
        sims_nameroot = "realgal",
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
        
        :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 15.0``.
        
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
        
        :returns: ``out_mef`` is the list containing the simulated images and ``out_true_cats`` is the list containing the information of the mock catalogs (hence ``len(out_mef) = len(out_true_cats) = nsimimages``).
        :rtype: list, list
            
        
        This method calls the routine ``comest.SrcPlacer.RealGalLocator`` to put the fake sources on this image. In this case it is the real galaxies observed catalog provided by **GalSim** and **COSMOS** teams. Please note that since we are resampling from the observed catalog, hence the configurations of the galaxy shape of ``hlr_dict``, ``fbulge_dict`` and ``q_dict`` do _NOT_ apply on this set of simulation. But for the sake of consistency,this routine still requires these input configuration. The simulated sources are uniformly distributed in the CCD with the number density of ``ngals_arcmin2``.
            
        .. seealso:: ``comest.SrcPlacer.RealGalLocator`` for more details about the configuration.
            
        """
        # assign the value and sanitize
        zeropoint   =   self.img_zp     # zeropoint
        # assign psf
        if  psf_dict  is None:
            psf_dict    =   {"moffat":{ "beta": 4.5, "fwhm": self.img_fwhm } }
            
        # diagnostic output
        print
        print "#", "Using RealGalLocator to put galaxies (bulge + disk) on the targeted image..."
        print "#", "path2image:", path2image
        print "#", "psf_dict:", psf_dict
        print "#", "stamp_size_arcsec:", stamp_size_arcsec
        print "#", "mag_dict:", mag_dict
        print "#", "hlr_dict:", hlr_dict
        print "#", "fbulge_dict:", fbulge_dict
        print "#", "q_dict:", q_dict
        print "#", "pos_ang_dict:", pos_ang_dict
        print "#", "ngals_arcmin2:", ngals_arcmin2
        print "#", "nsimimages:", nsimimages
        print "#", "random_seed:", random_seed
        print "#", "sims_nameroot:", sims_nameroot
        print "#", "ncpu:", ncpu
        print
        
        # timing
        t1      =       time.time()
        
        # run it and save it as the file.
        out_mef, out_true_cats  = SrcPlacer.RealGalLocator(
            path2image          = path2image,
            zeropoint           = zeropoint,
            psf_dict            = psf_dict,
            stamp_size_arcsec   = stamp_size_arcsec,
            mag_dict            = mag_dict,
            hlr_dict            = hlr_dict,
            fbulge_dict         = fbulge_dict,
            q_dict              = q_dict,
            pos_ang_dict        = pos_ang_dict,
            ngals_arcmin2       = ngals_arcmin2,
            nsimimages          = nsimimages,
            random_seed         = random_seed,
            ncpu                = ncpu,
            )
            
        # timing
        t2      =       time.time()
            
        # diagnostic
        print
        print "#", "Total time takes:", t2 - t1
        print
        
        # make sure dir exists
        outdir_sims             = os.path.join(self.path2outdir, sims_nameroot + "_sims")
        if os.path.isdir( outdir_sims ) == False: os.makedirs(outdir_sims)
        # save
        # for mef image
        SrcPlacer.galsim.fits.writeMulti(out_mef, outdir_sims + "/" + sims_nameroot + ".sims.fits")
        # for mef table
        phdulist = pyfits.PrimaryHDU( header = pyfits.BinTableHDU(data = out_true_cats[0]).header ) # primary table
        list4fits= [phdulist]                                                                       # put it in the list first
        for ntable in  xrange(nsimimages):                                                          # append all the tables
            list4fits.append( pyfits.BinTableHDU(data = out_true_cats[ntable]) )                    # append all the tables
        thdulist = pyfits.HDUList(list4fits)                                                        # convert it into table hdulist
        thdulist.writeto(outdir_sims + "/" + sims_nameroot + ".sims.cat.fits", clobber = True)      # save
        
        # diagnostic
        print
        print "#", "sims image:",
        print outdir_sims + "/" + sims_nameroot + ".sims.fits"
        print "#", "sims cat:",
        print outdir_sims + "/" + sims_nameroot + ".sims.cat.fits"
        print
        
        return out_mef, out_true_cats

    # ---
    # PntSrcLocator
    # ---
    def PntSrcLocator(self,
        path2image,
        psf_dict   = None,
        stamp_size_arcsec   =   20.0,
        mag_dict   = {"lo":20.0, "hi":25.0 },
        hlr_dict   = {"lo":0.35 , "hi":0.75  },
        fbulge_dict= {"lo":0.5 , "hi":0.9  },
        q_dict     = {"lo":0.4 , "hi":1.0  },
        pos_ang_dict={"lo":0.0 , "hi":180.0},
        ngals_arcmin2 = 15.0,
        nsimimages    = 50,
        random_seed   = 2342221,
        sims_nameroot = "pntsrc",
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
        
        :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 15.0``.
        
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
        
        :returns: ``out_mef`` is the list containing the simulated images and ``out_true_cats`` is the list containing the information of the mock catalogs (hence ``len(out_mef) = len(out_true_cats) = nsimimages``).
        :rtype: list, list
            
            
        This method calls the routine ``comest.SrcPlacer.PntSrcLocator`` to put the fake sources on this image. In this case it is for the point sources (e.g., stars) convolving with the given PSF. Please note that since we are simulating point sources, the actually shape of simulated sources is just the PSF with the size of given ``img_fwhm``. Therefore the configurations of ``hlr_dict``, ``fbulge_dict``, ``q_dict`` and ``pos_ang_dict`` do _NOT_ apply here. But for the sake of consistency,this routine still requires these input configuration. The simulated sources are uniformly distributed in the CCD with the number density of ``ngals_arcmin2``.
            
        .. seealso:: ``comest.SrcPlacer.PntSrcLocator`` for more details about the configuration.
            
        """
        # assign the value and sanitize
        zeropoint   =   self.img_zp     # zeropoint
        # assign psf
        if  psf_dict  is None:
            psf_dict    =   {"moffat":{ "beta": 4.5, "fwhm": self.img_fwhm } }
            
        # diagnostic output
        print
        print "#", "Using PntSrcLocator to put point sources on the targeted image..."
        print "#", "path2image:", path2image
        print "#", "psf_dict:", psf_dict
        print "#", "stamp_size_arcsec:", stamp_size_arcsec
        print "#", "mag_dict:", mag_dict
        print "#", "hlr_dict:", hlr_dict
        print "#", "fbulge_dict:", fbulge_dict
        print "#", "q_dict:", q_dict
        print "#", "pos_ang_dict:", pos_ang_dict
        print "#", "ngals_arcmin2:", ngals_arcmin2
        print "#", "nsimimages:", nsimimages
        print "#", "random_seed:", random_seed
        print "#", "sims_nameroot:", sims_nameroot
        print "#", "ncpu:", ncpu
        print
        
        # timing
        t1      =       time.time()
        
        # run it and save it as the file.
        out_mef, out_true_cats  = SrcPlacer.PntSrcLocator(
            path2image          = path2image,
            zeropoint           = zeropoint,
            psf_dict            = psf_dict,
            stamp_size_arcsec   = stamp_size_arcsec,
            mag_dict            = mag_dict,
            hlr_dict            = hlr_dict,
            fbulge_dict         = fbulge_dict,
            q_dict              = q_dict,
            pos_ang_dict        = pos_ang_dict,
            ngals_arcmin2       = ngals_arcmin2,
            nsimimages          = nsimimages,
            random_seed         = random_seed,
            ncpu                = ncpu,
            )
            
        # timing
        t2      =       time.time()
            
        # diagnostic
        print
        print "#", "Total time takes:", t2 - t1
        print
            
        # make sure dir exists
        outdir_sims             = os.path.join(self.path2outdir, sims_nameroot + "_sims")
        if os.path.isdir( outdir_sims ) == False: os.makedirs(outdir_sims)
        # save
        # for mef image
        SrcPlacer.galsim.fits.writeMulti(out_mef, outdir_sims + "/" + sims_nameroot + ".sims.fits")
        # for mef table
        phdulist = pyfits.PrimaryHDU( header = pyfits.BinTableHDU(data = out_true_cats[0]).header ) # primary table
        list4fits= [phdulist]                                                                       # put it in the list first
        for ntable in  xrange(nsimimages):                                                          # append all the tables
            list4fits.append( pyfits.BinTableHDU(data = out_true_cats[ntable]) )                    # append all the tables
        thdulist = pyfits.HDUList(list4fits)                                                        # convert it into table hdulist
        thdulist.writeto(outdir_sims + "/" + sims_nameroot + ".sims.cat.fits", clobber = True)      # save
        
        # diagnostic
        print
        print "#", "sims image:",
        print outdir_sims + "/" + sims_nameroot + ".sims.fits"
        print "#", "sims cat:",
        print outdir_sims + "/" + sims_nameroot + ".sims.cat.fits"
        print
        
        return out_mef, out_true_cats



    # ---
    # Run SE on sims
    # ---
    def RunSEforSims(self, sims_nameroot, sims_sex_args = "", path2maskmap = None, outputcheckimage = False, tol_fwhm = 1.0):
        """

        :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**.
        
        :param sims_sex_args: The additional argument you want to pass to **SExtractor** to run on simulated image. It should be the same as ``full_sex_args`` which you used to detect/catalog all the sources in the first palce for the fair comparison.
        
        :param path2maskmap: The absolute path to the masking map. This masking map MUST be in the exactly same shape as the observed image (and hence the simulated image). **ComEst** will mask every objects detected in simulated image with the position within 1 pixel of the masked region. Every pixel in ``path2maskmap`` with value > 0 is considered to be masked.  Therefore, it is convenient to use the segmentation map outputted by **SExtractor** as an alternative mask map if you dont want the simulated objects to be considered as 'detected' in the position where there is an object in the input image. It is handy to use the segementation map of BnB image created in the early state as the masking map. By default, it is ``path2maskmap = None``.
        
        :param outputcheckimage: Whether or not **SExtractor** will output the check-images (as the diagnostic output) and save them in the disk. By default, it is ``outputcheckimage = False``.
        
        :param tol_fwhm: The multiplicative factor of ``img_fwhm`` used in matching the **SExtractor** to the mock catalog. By default, it is ``tol_fwhm = 1`` meaning we match all the **SExtractor** objects to the mock catalog within 1 ``img_fwhm``.
        
        :type sims_nameroot: str
        :type sims_sex_args: str
        :type path2maskmap: str
        :type outputcheckimage: bool
        :type tol_fwhm: float
        
        :returns: ``stdout``. The standard output status of the run. ``stdout = 0`` for successful run.
        :rtype: int

            
        This functions read in the MEF of the simulated image with the code name of ``sims_nameroot`` and run **SExtractor** on those. After running **SExtractor** on all simulated images designed by ``sims_nameroot``, this routine will match the mock catalog of the input and **SExtractor** output catalog. In the end, it will save files to the subdirectory designed by ``sims_nameroot``.
        
        Four saved files are saved in the subdirectory designed by ``sims_nameroot``:
            * merged true catalog: [all true], the merged catalog containing all the input in mock. It is called ``sims_nameroot + "_sims" + "/" + sims_nameroot + ".sims.sex.merged_true.cat.fits"``.
            * matched pairs: [matched true] + [matched se], the matched pairs containing all the input sources which have been detected by **SExtractor**. It is called ``sims_nameroot + "_sims" + "/" + sims_nameroot + ".sims.sex.matched_pairs.cat.fits"``.
            * unmatched src: [true] - [matched true], the input sources in the mock which are not detected by **SExtractor**. It is called ``sims_nameroot + "_sims" + "/" + sims_nameroot + ".sims.sex.unmatched.cat.fits"``.
            * ghost: [se] - [matched se], the sources detected by **SExtractor** but they are not in the input mock. That is, they are ghost and considered as the false detection. It is called ``sims_nameroot + "_sims" + "/" + sims_nameroot + ".sims.sex.ghost.cat.fits"``.

        """
        # dignostic
        print
        print "#", "Running SE for sims..."
        print "#", "sims_nameroot:", sims_nameroot, "[Important: the sims_nameroot has to match the nameroot you create sims.]"
        print "#", "sims_sex_args:", sims_sex_args
        print "#", "path2maskmap:", path2maskmap
        print "#", "outputcheckimage:", outputcheckimage
        print "#", "tol_fwhm:", tol_fwhm, "[the objects are considered a matched pair if within this factor of fwhm.]"
        print
        
        # get the outdir_sims and path2sims_img
        outdir_sims     =   os.path.join(self.path2outdir, sims_nameroot + "_sims")
        path2sims_img   =   os.path.join(outdir_sims, sims_nameroot + ".sims.fits")
        
        # read in the fits image to see how many extension of that MEF.
        readin_mef      =   pyfits.open(path2sims_img)
        Nmef            =   len(readin_mef)
        
        # read in mask file if any
        if path2maskmap is not None:
            # read in mask
            readin_mask =   pyfits.getdata(path2maskmap)
            # check the shape
            if  (self.y_npixels, self.x_npixels) != readin_mask.shape:
                print
                print "#", "(y_npixels, x_npixels):",  (self.y_npixels, self.x_npixels)
                print "#", "shape od path2maskmap:", readin_mask.shape
                print
                raise RuntimeError("The dimensions of path2maskmap is not the same as the image. Please check!")
            # assign coordinates
            x_coord_mesh, y_coord_mesh  =   np.meshgrid(
                                            np.linspace(0, self.x_npixels, self.x_npixels+1),
                                            np.linspace(0, self.y_npixels, self.y_npixels+1))
            # get the masked pixels - anything > 0 is considered as masked
            i_am_masked =   ( readin_mask > 0 )
    
    
        # diagnostic
        print
        print "#", "number of the extension in", path2sims_img, ":", Nmef
        print
        print "#", "Running SE on sims ..."
        print
        
        # construct the args
        put_me_in_se_args =   " -MAG_ZEROPOINT" + "    " + "%.3f" % self.img_zp   + \
                              " -SEEING_FWHM"   + "    " + "%.3f" % self.img_fwhm + \
                              " -PIXEL_SCALE"   + "    " + "%.3f" % self.img_pixel_scale + \
                              "    " + sims_sex_args
        
        # construct the command
        run_me  =   []
        for nn_mef  in xrange(Nmef):
            # construct the command
            run_me.append(
                SE_CMD_CREATOR(
                se_exec_is      = self.sex_exec,
                img_is          = path2sims_img,
                se_config_is    = self.sex_config,
                se_params_is    = self.sex_params,
                outdir_is       = outdir_sims,
                name_root_is    = outdir_sims + "/" + sims_nameroot + ".sims.sex.%i" % nn_mef,
                args_is         = put_me_in_se_args,
                outputcheckimage= outputcheckimage,
                next            = nn_mef) )

        
        # run it!
        # later on we can parallize it but not now.
        stdout  =   []
        for nn_mef in xrange(Nmef):
            # diagnostic
            print
            print "#", run_me[nn_mef]
            print
            # use temp to record stdout
            stdout_tmp      =       os.system(run_me[nn_mef])
            # put it back in stdout
            stdout.append(stdout_tmp)
            # cleam
            del stdout_tmp
        
        # diagnostic
        print
        print "#", "Running SE on sims is done..."
        print
        print "#", "Now it reads in the SE catalog and the truth catalog."
        print
        
        # read the truth fits and se output catalog
        true_cat    =   []
        se_out_cat  =   []
        for nn_mef in xrange(Nmef):
            # naming
            true_cat_temp     =   outdir_sims + "/" + sims_nameroot + ".sims.cat.fits"
            se_out_cat_temp   =   outdir_sims + "/" + sims_nameroot + ".sims.sex.%i" % nn_mef + ".cat.fits"
            
            # read in se catalog - it is more tricky if we do have mask map.
            if path2maskmap is None:
                # read in true_cat - add copy in the end is to avoid the over-open the files
                true_cat.append(   pyfits.getdata(true_cat_temp  , ext = nn_mef + 1).copy() )
                # read in se_out_cat - add copy in the end is to avoid the over-open the files
                se_out_cat.append( pyfits.getdata(se_out_cat_temp, ext = 2         ).copy() )
            else:
                # first get the distance of the se / true sources to all the masked points
                true_temp = pyfits.getdata(true_cat_temp  , ext = nn_mef + 1).copy()
                se_temp   = pyfits.getdata(se_out_cat_temp, ext = 2         ).copy()
                # match to the masked area
                ds_true   = py2dmatch.carte2dmatch(x1 = true_temp["x_true"],
                                                   y1 = true_temp["y_true"],
                                                   x2 = x_coord_mesh[ i_am_masked ],
                                                   y2 = y_coord_mesh[ i_am_masked ],
                                                   tol = None,
                                                   nnearest = 1)[2]
                ds_se     = py2dmatch.carte2dmatch(x1 = se_temp["XWIN_IMAGE"],
                                                   y1 = se_temp["YWIN_IMAGE"],
                                                   x2 = x_coord_mesh[ i_am_masked ],
                                                   y2 = y_coord_mesh[ i_am_masked ],
                                                   tol = None,
                                                   nnearest = 1)[2]
                # update - only take the se sources which have nearest neighbor with distance > 1
                true_cat.append(   pyfits.getdata(true_cat_temp  , ext = nn_mef + 1).copy()[ (ds_true > 1) ] )
                se_out_cat.append( pyfits.getdata(se_out_cat_temp, ext = 2         ).copy()[ (ds_se   > 1) ] )
                # clean
                del se_temp, true_temp, ds_se, ds_true
            # clean
            del true_cat_temp, se_out_cat_temp
        
        # diagnostic
        print
        print "#", "Read in SE output cat is done."
        print
        print "#", "Start matching."
        print
        
        # match true_cat and se_out_cat in cartesian coordinate.
        i_am_matched_in_se    =   []
        i_am_matched_in_true  =   []
        i_am_distance_in_pixel=   []
        for nn_mef in xrange(Nmef):
            # match using py2dmatch in x and y
            # using tol = fwhm
            """
            i_am_matched_in_true_temp, i_am_matched_in_se_temp, i_am_distance_in_pixel_temp        = \
                py2dmatch.carte2dmatch(
                x1 = true_cat[nn_mef]["x_true"],
                y1 = true_cat[nn_mef]["y_true"],
                x2 = se_out_cat[nn_mef]["XWIN_IMAGE"],
                y2 = se_out_cat[nn_mef]["YWIN_IMAGE"],
                tol= tol_fwhm * (self.img_fwhm / self.img_pixel_scale),
                nnearest=1)
            """
            i_am_matched_in_se_temp, i_am_matched_in_true_temp, i_am_distance_in_pixel_temp        = \
                py2dmatch.carte2dmatch(
                x2 = true_cat[nn_mef]["x_true"],
                y2 = true_cat[nn_mef]["y_true"],
                x1 = se_out_cat[nn_mef]["XWIN_IMAGE"],
                y1 = se_out_cat[nn_mef]["YWIN_IMAGE"],
                tol= tol_fwhm * (self.img_fwhm / self.img_pixel_scale),
                nnearest=1)
            # Consider the magnitude matching by using sigma_cliping
            # That is, the se magnitude and true magnitude has to be within 3 sigma 
            # to be considered as a mathced pair - this can prevent from the mismatch.
            # For now we use MAG_AUTO - mag_true.
            '''
            filtered_data   =   py2dmatch.sigma_clip(
                                data = se_out_cat[nn_mef]["MAG_AUTO"][ i_am_matched_in_se_temp ] - true_cat[nn_mef]["mag_true"][ i_am_matched_in_true_temp ],
                                sig  = 3.0,
                                iters= None,
                                cenfunc=lambda x:0.0,
                                varfunc=np.var,
                                axis=None, copy=True)
            returned_masked =   filtered_data.mask.copy()
            '''
            returned_masked =   py2dmatch.Adpative_sigma_clip(
                    mag1 = se_out_cat[nn_mef]["MAG_AUTO"][ i_am_matched_in_se_temp ],
                    mag2 = true_cat[  nn_mef]["mag_true"][ i_am_matched_in_true_temp ],
                    sig  = 5.0,
                    iters= 1,
                    cenfunc=lambda x: 0.0,
                    varfunc=np.var)
            # save them in the array.
            i_am_matched_in_true.append( np.copy(i_am_matched_in_true_temp[ ~returned_masked ]) )
            i_am_matched_in_se.append( np.copy(i_am_matched_in_se_temp[ ~returned_masked ]) )
            i_am_distance_in_pixel.append( np.copy(i_am_distance_in_pixel_temp[ ~returned_masked ]) )
            # clean
            del i_am_matched_in_se_temp, i_am_matched_in_true_temp, i_am_distance_in_pixel_temp, returned_masked
        
        # diagnostic
        print
        print "#", "Matching is done."
        print
        print "#", "Outputing."
        print
        
        #
        # now we have the following catalogs
        # 1. merged true catalog: [all true]
        # 1. matched pairs: [matched true] + [matched se]
        # 2. unmatched src: [true] - [matched true]
        # 3. ghost: [se] - [matched se]
        #
        
        # define the file name
        merged_true_ff      =   outdir_sims + "/" + sims_nameroot + ".sims.sex.merged_true.cat.fits"
        matched_pairs_ff    =   outdir_sims + "/" + sims_nameroot + ".sims.sex.matched_pairs.cat.fits"
        unmatched_true_ff   =   outdir_sims + "/" + sims_nameroot + ".sims.sex.unmatched.cat.fits"
        ghost_ff            =   outdir_sims + "/" + sims_nameroot + ".sims.sex.ghost.cat.fits"
        
        # merge them
        merged_true          =   true_cat[0]
        merged_matched_pairs =   rfn.merge_arrays(
                                [ true_cat[0][ i_am_matched_in_true[0] ], se_out_cat[0][ i_am_matched_in_se[0] ] ],
                                flatten = True )
        merged_unmatched_true=   true_cat[0][ list( set( xrange( len(true_cat[0]) ) ) - set( i_am_matched_in_true[0] ) ) ]
        merged_ghost         =   se_out_cat[0][ list( set( xrange( len(se_out_cat[0]) ) ) - set( i_am_matched_in_se[0] ) ) ]
        for nn_mef in xrange(1,Nmef):
            merged_true             =   np.append(merged_true, true_cat[nn_mef])
            merged_matched_pairs    =   np.append(merged_matched_pairs,
                rfn.merge_arrays(
                [ true_cat[nn_mef][ i_am_matched_in_true[nn_mef] ], se_out_cat[nn_mef][ i_am_matched_in_se[nn_mef] ] ],
                flatten = True ) )
            merged_unmatched_true   =   np.append(merged_unmatched_true,
                true_cat[nn_mef][ list( set( xrange( len(true_cat[nn_mef]) ) ) - set( i_am_matched_in_true[nn_mef] ) ) ] )
            merged_ghost            =   np.append(merged_ghost,
                se_out_cat[nn_mef][ list( set( xrange( len(se_out_cat[nn_mef]) ) ) - set( i_am_matched_in_se[nn_mef] ) ) ] )

        # save tables
        tbhdu_merged_true   =   pyfits.BinTableHDU(data = merged_true)
        tbhdu_merged_pairs  =   pyfits.BinTableHDU(data = merged_matched_pairs)
        tbhdu_unmatched_true=   pyfits.BinTableHDU(data = merged_unmatched_true)
        tbhdu_ghost         =   pyfits.BinTableHDU(data = merged_ghost)

        tbhdu_merged_true.writeto(merged_true_ff, clobber = True)
        tbhdu_merged_pairs.writeto(matched_pairs_ff, clobber = True)
        tbhdu_unmatched_true.writeto(unmatched_true_ff, clobber = True)
        tbhdu_ghost.writeto(ghost_ff, clobber = True)

        # diagnostic output
        print
        print "#", "Output cats:"
        print "#", "merged true catalog [all true]:", merged_true_ff
        print "#", "matched pairs [matched true] + [matched se]:", matched_pairs_ff
        print "#", "unmatched src [true] - [matched true]:", unmatched_true_ff
        print "#", "ghost [se] - [matched se]:", ghost_ff
        print

        # return
        return 0
        #return true_cat, se_out_cat, merged_matched_pairs


    # derive the completeness as the function of magnitude and ccd position
    def DeriveCom(self, sims_nameroot, x_steps_arcmin = None, y_steps_arcmin = None, mag_edges = None, save_files = False):
        """
            
        :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**.
        
        :param x_steps_arcmin: The binning step in the x direction. It is in the unit of arcmin. Default is None, it uses 1 arcmin.
        
        :param y_steps_arcmin: The binning step in the y direction. It is in the unit of arcmin. Default is None, it uses 1 arcmin.
        
        :param mag_edges: The magnitude binning edges in deriving the completeness as a function of magnitude. Default is None, it use np.linspace(minimum of magnitude, maximum of magnitude, 0.1).
        
        :param save_files: Whether or not to save the files in the subdirectory assigned by ``sims_nameroot``.
        
        :type sims_nameroot: str
        :type x_steps_arcmin: float
        :type x_steps_arcmin: float
        :type map_edges: array
        :type save_files: bool
            
        :returns: A dictionary containing ``fcom_of_mag``, ``fcom_of_xy``, ``mag_bins``, ``x_edges`` and ``y_edges``}. ``fcom_of_mag`` is the completeness as a function of magnitude with the binning of  ``mag_edges``. It is in the same shape of ``mag_bins`` (the center of the bin), ``mag_bins`` is the center of the bin with the length of ``len(mag_edges) - 1``. ``fcom_of_xy`` is the completeness function as a function of position for a given magnitude cut in ``mag_bins``. Its shape is [length of ``mag_bin``, length of y bins, length of x bins]. ``x_edges`` is the binning edges in the unit of pixel for the binning in the x direction, while ``y_edges``is for the y direction.
        :rtype: dict
        
        
        This subroutine derives the completeness as a function of magnitude and image position.
        
        For deriving the completeness as a function of magnitude, it uses the ``mag_edges`` as the binning edges to do the histogram. For deriving the completeness as a funciton of the image position, it bins in x direction in the unit of ``x_steps_arcmin`` (in the unit of arcmin). Same thing applies to y-direction.
        
        .. note:: If one wants to have the reliable measurement for the completeness as a function of CCD position, it is _extremely_ important to enlarge the number of the simulated images ``nsimimages`` in the simulation run. Or one should use ``mag_edges`` with wider magnitude bin. Otherwise the poisson noise of the input mock is too large for this measurement to be useful.
        
        .. todo:: Possible to add the arguments for the sigma clipping when we match in the magnitude space.
        
        """
        # diagnostic
        print
        print "#", "sims_nameroot:", sims_nameroot
        print "#", "x_steps:", x_steps_arcmin, "[arcmin]"
        print "#", "y_steps:", y_steps_arcmin, "[arcmin]"
        print "#", "mag_edges:", mag_edges
        print
    
        # decide the path2catalog_sims
        outdir_sims                 =   os.path.join(self.path2outdir, sims_nameroot + "_sims")
        path2true_cat               =   outdir_sims + "/" + sims_nameroot + ".sims.sex.merged_true.cat.fits"
        path2matched_pairs_cat      =   outdir_sims + "/" + sims_nameroot + ".sims.sex.matched_pairs.cat.fits"
    
        # read in file
        true_cat                =   pyfits.getdata(path2true_cat)
        matched_pairs_cat       =   pyfits.getdata(path2matched_pairs_cat)
    
        # decide binning in magnitude
        if mag_edges is None:
            mag_edges           =   np.arange( true_cat["mag_true"].min(), true_cat["mag_true"].max() + 0.1, 0.1 )
        # derive mag_bins
        mag_bins                =   0.5 * (mag_edges[1:] + mag_edges[:-1])
        
        # decide binning in x and y
        if x_steps_arcmin is None:   x_steps_arcmin   =   1.0
        if y_steps_arcmin is None:   y_steps_arcmin   =   1.0

        # image binning
        x_steps = int( x_steps_arcmin * 60.0 / self.img_pixel_scale )
        x_edges = np.linspace( 0.5, self.x_npixels + 0.5, int(self.x_npixels / x_steps) + 1 )
        x_bins  = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_steps = int( y_steps_arcmin * 60.0 / self.img_pixel_scale )
        y_edges = np.linspace( 0.5, self.y_npixels + 0.5, int(self.y_npixels / y_steps) + 1 )
        y_bins  = 0.5 * (y_edges[1:] + y_edges[:-1])
    

        # extract information from cat
        true_dict                   =   {
            "mag": true_cat["mag_true"],
            "x"  : true_cat["x_true"],
            "y"  : true_cat["y_true"],
            }
        matched_paris_dict          =   {
            "mag": matched_pairs_cat["mag_true"],
            "x"  : matched_pairs_cat["x_true"],
            "y"  : matched_pairs_cat["y_true"],
            }

        # hist
        true_dict.update({          "hist1d" :   np.histogram(true_dict["mag"]         , bins = mag_edges)[0] })
        matched_paris_dict.update({ "hist1d" :   np.histogram(matched_paris_dict["mag"], bins = mag_edges)[0] })
    
        # hist2d
        true_dict.update({          "hist2d" :   np.array([
                                                 np.histogram2d(
                                                 x = true_dict["x"][ ( true_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                     ( true_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 y = true_dict["y"][ ( true_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                     ( true_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 bins = [x_edges, y_edges])[0]
                                                 for nmag in xrange(len(mag_bins)) ])
                         })
        matched_paris_dict.update({ "hist2d" :   np.array([
                                                 np.histogram2d(
                                                 x = matched_paris_dict["x"][ ( matched_paris_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                              ( matched_paris_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 y = matched_paris_dict["y"][ ( matched_paris_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                              ( matched_paris_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 bins = [x_edges, y_edges])[0]
                                                 for nmag in xrange(len(mag_bins)) ])
                        })
        # derive the completeness as the function of magnitude - fcom(mag)
        fcom_of_mag                 =   matched_paris_dict["hist1d"] * 1.0 / true_dict["hist1d"]
        #fcomerr_of_mag              =   np.hypot( np.sqrt(matched_paris_dict["hist1d"] * 1.0) , np.sqrt( true_dict["hist1d"] * 1.0 ) )
                     
        # derive the completeness as the function of pixel - fcom(x,y)
        fcom_of_xy                  =   matched_paris_dict["hist2d"] * 1.0 / true_dict["hist2d"]
        #fcomerr_of_xy               =   np.hypot( np.sqrt(matched_paris_dict["hist2d"] * 1.0) , np.sqrt( true_dict["hist2d"] * 1.0 ) )
        
        # save files?
        if  save_files == True:
            # save the result as a funciton of magnitude
            path2outmag     =   os.path.join(outdir_sims, sims_nameroot + ".sims.fcommag.dat")
            # write the fcommag
            open2write      =   open(path2outmag, "w")
            # header
            open2write.write("#    mag_lo    mag_me    mag_hi    fcom    " + "\n")
            # content
            for nnm in xrange(len(mag_bins)):
                open2write.write("    " + "%.3f" % mag_edges[:-1][nnm] +
                                 "    " + "%.3f" % mag_bins[nnm]       +
                                 "    " + "%.3f" % mag_edges[1: ][nnm] +
                                 "    " + "%.3f" % fcom_of_mag[nnm]    +
                                 "    " + "\n"  )
            # record
            open2write.write("#    file created at " + time.ctime() + "\n")
            # close file
            open2write.close()
            
            # save the result as a function of position - this require reading in the simulated image.
            path2sims_img   =   os.path.join(outdir_sims, sims_nameroot + ".sims.fits")
            path2outmap     =   os.path.join(outdir_sims, sims_nameroot + ".sims.fcomxy.fits")
            path2outtruemap         =   os.path.join(outdir_sims, sims_nameroot + ".sims.nsrc_merged_true_xy.fits")
            path2outmatchedpairsmap =   os.path.join(outdir_sims, sims_nameroot + ".sims.nsrc_matched_pairs_xy.fits")
            # read in the fits image to see how many extension of that MEF.
            readinheader    =   pyfits.getheader(path2sims_img)
            # declare a list
            list_of_fcomxy  =   [ ]
            list_of_nsrc_merged_true_xy       =   []
            list_of_nsrc_matched_pairs_xy     =   []
            # change the values in mef
            for nnm in xrange(len(mag_bins)):
                # append header and update header
                temp_fits_header              =    readinheader.copy()
                temp_fits_header["NAXIS1"]    =    len(x_bins)
                temp_fits_header["NAXIS2"]    =    len(y_bins)
                #temp_fits_header["CRPIX1"]    =    len(x_bins) / 2.0 + 0.5
                #temp_fits_header["CRPIX2"]    =    len(y_bins) / 2.0 + 0.5
                temp_fits_header["CRPIX1"]    =    0.5 + ( temp_fits_header["CRPIX1"] - 0.5 ) / ( x_edges[1] - x_edges[0] )
                temp_fits_header["CRPIX2"]    =    0.5 + ( temp_fits_header["CRPIX2"] - 0.5 ) / ( y_edges[1] - y_edges[0] )
                temp_fits_header["CD1_1" ]    =    temp_fits_header["CD1_1" ] * ( x_edges[1] - x_edges[0] )
                temp_fits_header["CD2_2" ]    =    temp_fits_header["CD2_2" ] * ( y_edges[1] - y_edges[0] )
                
                temp_fits_header.append(
                    pyfits.Card('MAG_LO', mag_edges[:-1][nnm], 'the mag_lo of the magnitude slice of fcom_xy') )
                temp_fits_header.append(
                    pyfits.Card('MAG_ME', mag_bins[nnm]      , 'the mag_me of the magnitude slice of fcom_xy') )
                temp_fits_header.append(
                    pyfits.Card('MAG_HI', mag_edges[1: ][nnm], 'the mag_hi of the magnitude slice of fcom_xy') )
                # get the data
                #temp_fits_data      =   Nearest_interpolator(
                #                        values_array = fcom_of_xy[nnm], 
                #                        x_edges = x_edges, 
                #                        y_edges = y_edges, 
                #                        new_x_edges = np.append(0.5, np.arange(0,self.x_npixels) + 1 + 0.5 ), 
                #                        new_y_edges = np.append(0.5, np.arange(0,self.y_npixels) + 1 + 0.5 ) )
                temp_fits_data       =   np.copy( fcom_of_xy[nnm] )
                temp_fits_true       =   np.copy( true_dict["hist2d"][nnm] )
                temp_fits_matched    =   np.copy( matched_paris_dict["hist2d"][nnm] )
                # if it is the primary hdu or not
                if nnm == 0:
                    # append this fits
                    list_of_fcomxy.append( pyfits.PrimaryHDU(data = temp_fits_data, header = temp_fits_header) )
                    list_of_nsrc_merged_true_xy.append( pyfits.PrimaryHDU(data = temp_fits_true, header = temp_fits_header) )
                    list_of_nsrc_matched_pairs_xy.append( pyfits.PrimaryHDU(data = temp_fits_matched, header = temp_fits_header) )
                else:
                    # append this fits
                    list_of_fcomxy.append( pyfits.ImageHDU(  data = temp_fits_data, header = temp_fits_header) )
                    list_of_nsrc_merged_true_xy.append( pyfits.ImageHDU(  data = temp_fits_true, header = temp_fits_header) )
                    list_of_nsrc_matched_pairs_xy.append( pyfits.ImageHDU(  data = temp_fits_matched, header = temp_fits_header) )

            # make HDULIST
            output_mef          =   pyfits.HDUList( hdus = list_of_fcomxy )
            output_mef_true     =   pyfits.HDUList( hdus = list_of_nsrc_merged_true_xy )
            output_mef_matched  =   pyfits.HDUList( hdus = list_of_nsrc_matched_pairs_xy )
            # write output
            output_mef.writeto(path2outmap, clobber = True)
            output_mef_true.writeto(path2outtruemap, clobber = True)
            output_mef_matched.writeto(path2outmatchedpairsmap, clobber = True)
            # close file
            output_mef.close()
            output_mef_true.close()
            output_mef_matched.close()
            
            # diagnostic output
            print
            print "#", "output:", path2outmag
            print "#", "output:", path2outmap
            print "#", "output:", path2outtruemap
            print "#", "output:", path2outmatchedpairsmap
            print            
                     
        return {"fcom_of_mag": fcom_of_mag, "fcom_of_xy": fcom_of_xy, "mag_bins": mag_bins, "x_edges": x_edges, "y_edges": y_edges}



    # derive the purity as the function of magnitude and ccd position
    def DerivePur(self, sims_nameroot, x_steps_arcmin = None, y_steps_arcmin = None, mag_edges = None, save_files = False):
        """
            
        :param sims_nameroot: The code name you want to identify this run of simulation. It is not only the name of the subdirectory for saving the images simulated in this run, but also the code name for **ComEst** to identify the simulation for the remaining analysis pipeline. IMPORTANT: Please use the consistent code name ``sims_nameroot`` for this set of simulated images throughout **ComEst**.
        
        :param x_steps_arcmin: The binning step in the x direction. It is in the unit of arcmin. Default is None, it uses 1 arcmin.
        
        :param y_steps_arcmin: The binning step in the y direction. It is in the unit of arcmin. Default is None, it uses 1 arcmin.
        
        :param mag_edges: The magnitude binning edges in deriving the completeness as a function of magnitude. Default is None, it use np.linspace(minimum of magnitude, maximum of magnitude, 0.1).
        
        :param save_files: Whether or not to save the files in the subdirectory assigned by ``sims_nameroot``.
        
        :type sims_nameroot: str
        :type x_steps_arcmin: float
        :type x_steps_arcmin: float
        :type map_edges: array
        :type save_files: bool
            
        :returns: A dictionary containing ``fpur_of_mag``, ``fpur_of_xy``, ``mag_bins``, ``x_edges`` and ``y_edges``}. ``fpur_of_mag`` is the purity as a function of magnitude with the binning of  ``mag_edges``. It is in the same shape of ``mag_bins`` (the center of the bin), ``mag_bins`` is the center of the bin with the length of ``len(mag_edges) - 1``. ``fpur_of_xy`` is the purity function as a function of position for a given magnitude cut in ``mag_bins``. Its shape is [length of ``mag_bin``, length of y bins, length of x bins]. ``x_edges`` is the binning edges in the unit of pixel for the binning in the x direction, while ``y_edges``is for the y direction.
        :rtype: dict
            
            
        This subroutine derives the purity as a function of magnitude and image position.
            
        For deriving the purity as a function of magnitude, it uses the ``mag_edges`` as the binning edges to do the histogram. For deriving the purity as a funciton of the image position, it bins in x direction in the unit of ``x_steps_arcmin`` (in the unit of arcmin). Same thing applies to y-direction. 
        
        The same routine applied in deriving completeness is applied here except the following difference. Since we derive the purity of the source detection by estimating how many sources detected by **SExtractor** which are _NOT_ in the input catalog, we use the ``MAG_AUTO``, ``XWIN_IMAGE`` and ``YWIN_IMAGE`` returned by **SExtractor** as the reference for the "ghost" objects.
        
            
        .. note:: If one wants to have the reliable measurement for the purity as a function of CCD position, it is _extremely_ important to enlarge the number of the simulated images ``nsimimages`` in the simulation run. Or one should use ``mag_edges`` with wider magnitude bin. Otherwise the poisson noise of the input mock is too large for this measurement to be useful.
        .. seealso:: ``comest.ComEst.DeriveCom``
        
        """
        # diagnostic
        print
        print "#", "sims_nameroot:", sims_nameroot
        print "#", "x_steps:", x_steps_arcmin, "[arcmin]"
        print "#", "y_steps:", y_steps_arcmin, "[arcmin]"
        print "#", "mag_edges:", mag_edges
        print
    
        # decide the path2catalog_sims
        outdir_sims                 =   os.path.join(self.path2outdir, sims_nameroot + "_sims")
        path2matched_pairs_cat      =   outdir_sims + "/" + sims_nameroot + ".sims.sex.matched_pairs.cat.fits"
        path2ghost_cat              =   outdir_sims + "/" + sims_nameroot + ".sims.sex.ghost.cat.fits"

        # read in file
        matched_pairs_cat       =   pyfits.getdata(path2matched_pairs_cat)
        ghost_cat               =   pyfits.getdata(path2ghost_cat)

        # decide binning in magnitude
        if mag_edges is None:
            mag_edges           =   np.arange( matched_pairs_cat["mag_true"].min(), matched_pairs_cat["mag_true"].max() + 0.1, 0.1 )
        # derive mag_bins
        mag_bins                =   0.5 * (mag_edges[1:] + mag_edges[:-1])
        
        # decide binning in x and y
        if x_steps_arcmin is None:   x_steps_arcmin   =   1.0
        if y_steps_arcmin is None:   y_steps_arcmin   =   1.0

        # image binning
        x_steps = int( x_steps_arcmin * 60.0 / self.img_pixel_scale )
        x_edges = np.linspace( 0.5, self.x_npixels + 0.5, int(self.x_npixels / x_steps) + 1 )
        x_bins  = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_steps = int( y_steps_arcmin * 60.0 / self.img_pixel_scale )
        y_edges = np.linspace( 0.5, self.y_npixels + 0.5, int(self.y_npixels / y_steps) + 1 )
        y_bins  =   0.5 * (y_edges[1:] + y_edges[:-1])
    

        # extract information from cat
        # for now we use mag_auto, we can change it later if we want.
        # for the matched_pairs, we use mag_true since it is the *true* input and free from the photometry scatter.
        # for now we use WIN_IMAGE, we can change it later if we want.
        matched_pairs_dict                   =   {
            "mag": matched_pairs_cat["MAG_AUTO"],
            "x"  : matched_pairs_cat["XWIN_IMAGE"],
            "y"  : matched_pairs_cat["YWIN_IMAGE"],
            }
        ghost_dict          =   {
            "mag": ghost_cat["MAG_AUTO"],
            "x"  : ghost_cat["XWIN_IMAGE"],
            "y"  : ghost_cat["YWIN_IMAGE"],
            }

        # hist
        matched_pairs_dict.update({          "hist1d" :   np.histogram(matched_pairs_dict["mag"], bins = mag_edges)[0] })
        ghost_dict.update({                  "hist1d" :   np.histogram(ghost_dict["mag"]        , bins = mag_edges)[0] })
    
        # hist2d
        matched_pairs_dict.update({ "hist2d" :   np.array([
                                                 np.histogram2d(
                                                 x = matched_pairs_dict["x"][ ( matched_pairs_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                              ( matched_pairs_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 y = matched_pairs_dict["y"][ ( matched_pairs_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                              ( matched_pairs_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 bins = [x_edges, y_edges])[0]
                                                 for nmag in xrange(len(mag_bins)) ])
                         })
        ghost_dict.update({         "hist2d" :   np.array([
                                                 np.histogram2d(
                                                 x = ghost_dict["x"][ ( ghost_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                      ( ghost_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 y = ghost_dict["y"][ ( ghost_dict["mag"] >= mag_edges[:-1][nmag] ) & \
                                                                      ( ghost_dict["mag"] <= mag_edges[1: ][nmag] ) ],
                                                 bins = [x_edges, y_edges])[0]
                                                 for nmag in xrange(len(mag_bins)) ])
                        })

        # derive the purity as the function of magnitude - fpur(mag)
        fpur_of_mag                 =   matched_pairs_dict["hist1d"] * 1.0 / ( matched_pairs_dict["hist1d"] + ghost_dict["hist1d"] )
        #fpur_of_mag[ (fpur_of_mag < 0) ] = 0.0
        #fpurerr_of_mag              =   np.hypot( np.sqrt(ghost_dict["hist1d"] * 1.0) , np.sqrt( true_dict["hist1d"] * 1.0 ) )
                     
        # derive the purity as the function of pixel - fpur(x,y)
        fpur_of_xy                  =   matched_pairs_dict["hist2d"] * 1.0 / ( matched_pairs_dict["hist2d"] + ghost_dict["hist2d"] )
        #fpur_of_xy[  (fpur_of_xy < 0 ) ] = 0.0
        #fpurerr_of_xy               =   np.hypot( np.sqrt(ghost_dict["hist2d"] * 1.0) , np.sqrt( true_dict["hist2d"] * 1.0 ) )


        # save files?
        if  save_files == True:
            # save the result as a funciton of magnitude
            path2outmag     =   os.path.join(outdir_sims, sims_nameroot + ".sims.fpurmag.dat")
            # write the fcommag
            open2write      =   open(path2outmag, "w")
            # header
            open2write.write("#    mag_lo    mag_me    mag_hi    fpur    " + "\n")
            # content
            for nnm in xrange(len(mag_bins)):
                open2write.write("    " + "%.3f" % mag_edges[:-1][nnm] +
                                 "    " + "%.3f" % mag_bins[nnm]       +
                                 "    " + "%.3f" % mag_edges[1: ][nnm] +
                                 "    " + "%.3f" % fpur_of_mag[nnm]    +
                                 "    " + "\n"  )
            # record
            open2write.write("#    file created at " + time.ctime() + "\n")
            # close file
            open2write.close()
            
            # save the result as a function of position - this require reading in the simulated image.
            path2sims_img   =   os.path.join(outdir_sims, sims_nameroot + ".sims.fits")
            path2outmap     =   os.path.join(outdir_sims, sims_nameroot + ".sims.fpurxy.fits")
            path2outallse   =   os.path.join(outdir_sims, sims_nameroot + ".nsrc_allse_xy.fits")
            # read in the fits image to see how many extension of that MEF.
            readinheader    =   pyfits.getheader(path2sims_img)
            # declare a list
            list_of_fpurxy  =   [ ]
            list_of_allse   =   [ ]
            # change the values in mef
            for nnm in xrange(len(mag_bins)):
                # append header and update header
                temp_fits_header              =    readinheader.copy()
                temp_fits_header["NAXIS1"]    =    len(x_bins)
                temp_fits_header["NAXIS2"]    =    len(y_bins)
                #temp_fits_header["CRPIX1"]    =    len(x_bins) / 2.0 + 0.5
                #temp_fits_header["CRPIX2"]    =    len(y_bins) / 2.0 + 0.5
                temp_fits_header["CRPIX1"]    =    0.5 + ( temp_fits_header["CRPIX1"] - 0.5 ) / ( x_edges[1] - x_edges[0] )
                temp_fits_header["CRPIX2"]    =    0.5 + ( temp_fits_header["CRPIX2"] - 0.5 ) / ( y_edges[1] - y_edges[0] )
                temp_fits_header["CD1_1" ]    =    temp_fits_header["CD1_1" ] * ( x_edges[1] - x_edges[0] )
                temp_fits_header["CD2_2" ]    =    temp_fits_header["CD2_2" ] * ( y_edges[1] - y_edges[0] )
                
                temp_fits_header.append(
                    pyfits.Card('MAG_LO', mag_edges[:-1][nnm], 'the mag_lo of the magnitude slice of fpur_xy') )
                temp_fits_header.append(
                    pyfits.Card('MAG_ME', mag_bins[nnm]      , 'the mag_me of the magnitude slice of fpur_xy') )
                temp_fits_header.append(
                    pyfits.Card('MAG_HI', mag_edges[1: ][nnm], 'the mag_hi of the magnitude slice of fpur_xy') )
                # get the data
                #temp_fits_data      =   Nearest_interpolator(
                #                        values_array = fput_of_xy[nnm], 
                #                        x_edges = x_edges, 
                #                        y_edges = y_edges, 
                #                        new_x_edges = np.append(0.5, np.arange(0,self.x_npixels) + 1 + 0.5 ), 
                #                        new_y_edges = np.append(0.5, np.arange(0,self.y_npixels) + 1 + 0.5 ) )
                temp_fits_data       =   np.copy( fpur_of_xy[nnm] )
                temp_fits_allse      =   np.copy( matched_pairs_dict["hist2d"][nnm] + ghost_dict["hist2d"][nnm] )
                # if it is the primary hdu or not
                if nnm == 0:
                    # append this fits
                    list_of_fpurxy.append( pyfits.PrimaryHDU(data = temp_fits_data, header = temp_fits_header) )
                    list_of_allse.append( pyfits.PrimaryHDU(data = temp_fits_allse, header = temp_fits_header) )
                else:
                    # append this fits
                    list_of_fpurxy.append( pyfits.ImageHDU(  data = temp_fits_data, header = temp_fits_header) )
                    list_of_allse.append( pyfits.ImageHDU(  data = temp_fits_allse, header = temp_fits_header) )
            
            # make HDULIST
            output_mef        =   pyfits.HDUList( hdus = list_of_fpurxy )
            output_mef_allse  =   pyfits.HDUList( hdus = list_of_allse  )
            # write output
            output_mef.writeto(path2outmap, clobber = True)
            output_mef_allse.writeto(path2outallse, clobber = True)
            # close file
            output_mef.close()
            output_mef_allse.close()
            
            # diagnostic output
            print
            print "#", "output:", path2outmag
            print "#", "output:", path2outmap
            print "#", "output:", path2outallse
            print
            
                        
        return {"fpur_of_mag": fpur_of_mag, "fpur_of_xy": fpur_of_xy, "mag_bins": mag_bins, "x_edges": x_edges, "y_edges": y_edges}

