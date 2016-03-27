#############
#
# Match two sets of data in the 2d plane.
# I.e., find nearest neighbor of one that's in the other.
# This routine is modified from some scripts I found online, forgot whom I looked for...
# Please let me know if you are the author.
#
# I copy the astropy.stats.sigma_clipping in this module for sigma clipping.
# see webpage: https://astropy.readthedocs.org/en/v1.0.5/_modules/astropy/stats/sigma_clipping.html#sigma_clip
#
#############

import numpy as np
from math import *

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

def carte2dmatch(x1, y1, x2, y2, tol= None, nnearest=1):
    """
    Finds matches in one catalog to another.

    Parameters
    x1 : array-like
        Cartesian coordinate x of the first catalog
    y1 : array-like
        Cartesian coordinate y of the first catalog (shape of array must match `x1`)
    x2 : array-like
        Cartesian coordinate x of the second catalog
    y2 : array-like
        Cartesian coordinate y of the second catalog (shape of array must match `x2`)
    tol : float or None, optional
        How close (in the unit of the cartesian coordinate) a match has to be to count as a match.  If None,
        all nearest neighbors for the first catalog will be returned.
    nnearest : int, optional
        The nth neighbor to find.  E.g., 1 for the nearest nearby, 2 for the
        second nearest neighbor, etc.  Particularly useful if you want to get
        the nearest *non-self* neighbor of a catalog.  To do this, use:
        ``carte2dmatch(x, y, x, y, nnearest=2)``

    Returns
    -------
    idx1 : int array
        Indecies into the first catalog of the matches. Will never be
        larger than `x1`/`y1`.
    idx2 : int array
        Indecies into the second catalog of the matches. Will never be
        larger than `x1`/`y1`.
    ds : float array
        Distance (in the unit of the cartesian coordinate) between the matches



    """
    # sanitize
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)

    # check
    if x1.shape != y1.shape:
        raise ValueError('x1 and y1 do not match!')
    if x2.shape != y2.shape:
        raise ValueError('x2 and y2 do not match!')

    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1

    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    # set kdt for coord2
    kdt = KDT(coords2)
    if nnearest == 1:
        idxs2 = kdt.query(coords1)[1]
    elif nnearest > 1:
        idxs2 = kdt.query(coords1, nnearest)[1][:, -1]
    else:
        raise ValueError('invalid nnearest ' + str(nnearest))

    # calc the distance
    ds = np.hypot(x1 - x2[idxs2], y1 - y2[idxs2])

    # index for coord1
    idxs1 = np.arange(x1.size)

    if tol is not None:
        msk = ds < tol
        idxs1 = idxs1[msk]
        idxs2 = idxs2[msk]
        ds = ds[msk]

    return idxs1, idxs2, ds


# ---
# 3d match
# ---
def carte2d_and_z_match(x1, y1, z1, x2, y2, z2, ztol, stol):
    """
    Finds matches in one catalog to another.

    Parameters
    x1 : array-like
        Cartesian coordinate x of the first catalog
    y1 : array-like
        Cartesian coordinate y of the first catalog (shape of array must match `x1`)
    z1 : array-like
        Cartesian coordinate z of the first catalog (shape of array must match `x1`)
    x2 : array-like
        Cartesian coordinate x of the second catalog
    y2 : array-like
        Cartesian coordinate y of the second catalog (shape of array must match `x2`)
    z2 : array-like
        Cartesian coordinate z of the second catalog (shape of array must match `x2`)
    ztol: float or array-like
        The tolarance in z direction. Its shape must match to `x1` if it is an array.
    stol: float or None, optional
        How close (in the unit of the cartesian coordinate) a match has to be to count as a match.  If None,
        all nearest neighbors for the first catalog will be returned.
    nnearest : int, optional
        The nth neighbor to find.  E.g., 1 for the nearest nearby, 2 for the
        second nearest neighbor, etc.  Particularly useful if you want to get
        the nearest *non-self* neighbor of a catalog.  To do this, use:
        ``carte2dmatch(x, y, x, y, nnearest=2)``

    Returns
    -------
    idx1 : int array
        Indecies into the first catalog of the matches. Will never be
        larger than `x1`/`y1`.
    idx2 : int array
        Indecies into the second catalog of the matches. Will never be
        larger than `x1`/`y1`.
    ds : float array
        Distance (in the unit of the cartesian coordinate) between the matches
    dz : float array
        Distance (in the unit of the cartesian coordinate) between the matches


    """
    # sanitize
    x1   = np.array(x1, copy=False)
    y1   = np.array(y1, copy=False)
    z1   = np.array(z1, copy=False)
    x2   = np.array(x2, copy=False)
    y2   = np.array(y2, copy=False)
    z2   = np.array(z2, copy=False)

    # check
    if x1.shape != y1.shape or x1.shape != z1.shape:
        raise ValueError('x1 and y1/z1 do not match!')
    if x2.shape != y2.shape or x2.shape != z2.shape:
        raise ValueError('x2 and y2/z2 do not match!')

    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1

    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    # set kdt for coord2
    kdt = KDT(coords2)

    # ---
    # Match using kdt
    # ---
    idxs2_within_balls   = kdt.query_ball_point(coords1, stol)                    # find the neighbors within a ball
    n_within_ball        = np.array(map(len, idxs2_within_balls), dtype = np.int) # counts within each ball
    zero_within_ball     = np.where( n_within_ball == 0)[0]                       # find which one does not have neighbors
    nonzero_within_ball  = np.where( n_within_ball >  0)[0]                       # find which one has neighbors
    
    # declare the distance / idxs2 for each element in nonzero_within_ball
    # I use no-brain looping here, slow but seems to be acceptable
    dz_within_ball       = []   # the distance
    idxs2                = []
    for i in nonzero_within_ball:
        #print i, len(idxs2_within_balls[i]), z1[i], z2[ idxs2_within_balls[i] ]
        # Another sub-kdt within a ball, but this times we use kdt.query to find the nearest one
        dz_temp, matched_id_temp   = KDT( np.transpose([ z2[ idxs2_within_balls[i] ] ]) ).query( np.transpose([ z1[i] ]) )
        matched_id_temp            = idxs2_within_balls[i][ matched_id_temp ]
        # append
        dz_within_ball.append(dz_temp)      # the distance of the nearest neighbor within the ball
        idxs2.append(matched_id_temp)       # the index in array2 of the nearest neighbor within the ball

    # index for coord1 - only using the object with non-zero neighbor in the ball
    idxs1           = np.arange(x1.size)[ nonzero_within_ball ]
    idxs2           = np.array(idxs2, dtype = np.int)
    dz_within_ball  = np.array(dz_within_ball, dtype = np.float)

    # clean
    del dz_temp, matched_id_temp

    # msk to clean the object with dz > ztol
    ztol            = np.array(ztol, ndmin=1)
    if    len(ztol) ==  1:
        msk             = ( dz_within_ball < ztol )
    elif  len(ztol) ==  len(x1):
        msk             = ( dz_within_ball < ztol[ nonzero_within_ball ] )
    else:
        raise ValueError("The length of ztol has to be 1 (float) or as the same as input x1/y1. len(ztol):", len(ztol))

    # only keep the matches which have dz < ztol
    idxs1           = idxs1[ msk ]
    idxs2           = idxs2[ msk ]
    ds              = np.hypot( x1[idxs1] - x2[idxs2], y1[idxs1] - y2[idxs2] )
    dz              = dz_within_ball[ msk ]
    
    return idxs1, idxs2, ds, dz



############################################################
#
# sigma clipping from  astropy.stats
# Under the LICENSE:
# Licensed under a 3-clause BSD style license
# 
############################################################

def sigma_clip(data, sig=3.0, iters=1, cenfunc=np.ma.median, varfunc=np.var,
               axis=None, copy=True):
    """Perform sigma-clipping on the provided data.

    This performs the sigma clipping algorithm - i.e. the data will be iterated
    over, each time rejecting points that are more than a specified number of
    standard deviations discrepant.

    .. note::
        `scipy.stats.sigmaclip
        <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sigmaclip.html>`_
        provides a subset of the functionality in this function.

    Parameters
    ----------
    data : array-like
        The data to be sigma-clipped (any shape).
    sig : float
        The number of standard deviations (*not* variances) to use as the
        clipping limit.
    iters : int or `None`
        The number of iterations to perform clipping for, or `None` to clip
        until convergence is achieved (i.e. continue until the last
        iteration clips nothing).
    cenfunc : callable
        The technique to compute the center for the clipping. Must be a
        callable that takes in a masked array and outputs the central value.
        Defaults to the median (numpy.median).
    varfunc : callable
        The technique to compute the standard deviation about the center. Must
        be a callable that takes in a masked array and outputs a width
        estimator::

             deviation**2 > sig**2 * varfunc(deviation)

        Defaults to the variance (numpy.var).

    axis : int or `None`
        If not `None`, clip along the given axis.  For this case, axis=int will
        be passed on to cenfunc and varfunc, which are expected to return an
        array with the axis dimension removed (like the numpy functions).
        If `None`, clip over all values.  Defaults to `None`.
    copy : bool
        If `True`, the data array will be copied.  If `False`, the masked array
        data will contain the same array as ``data``.  Defaults to `True`.

    Returns
    -------
    filtered_data : `numpy.ma.MaskedArray`
        A masked array with the same shape as ``data`` input, where the points
        rejected by the algorithm have been masked.

    Notes
    -----
     1. The routine works by calculating::

            deviation = data - cenfunc(data [,axis=int])

        and then setting a mask for points outside the range::

            data.mask = deviation**2 > sig**2 * varfunc(deviation)

        It will iterate a given number of times, or until no further points are
        rejected.

     2. Most numpy functions deal well with masked arrays, but if one would
        like to have an array with just the good (or bad) values, one can use::

            good_only = filtered_data.data[~filtered_data.mask]
            bad_only = filtered_data.data[filtered_data.mask]

        However, for multidimensional data, this flattens the array, which may
        not be what one wants (especially is filtering was done along an axis).

    Examples
    --------

    This will generate random variates from a Gaussian distribution and return
    a masked array in which all points that are more than 2 *sample* standard
    deviation from the median are masked::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, 2, 1)

    This will clipping on a similar distribution, but for 3 sigma relative to
    the sample *mean*, will clip until converged, and does not copy the data::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> from numpy import mean
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, 3, None, mean, copy=False)

    This will clip along one axis on a similar distribution with bad points
    inserted::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import normal
        >>> from numpy import arange, diag, ones
        >>> data = arange(5)+normal(0.,0.05,(5,5))+diag(ones(5))
        >>> filtered_data = sigma_clip(data, axis=0, sig=2.3)

    Note that along the other axis, no points would be masked, as the variance
    is higher.

    """

    if axis is not None:
        cenfunc_in = cenfunc
        varfunc_in = varfunc
        cenfunc = lambda d: np.expand_dims(cenfunc_in(d, axis=axis), axis=axis)
        varfunc = lambda d: np.expand_dims(varfunc_in(d, axis=axis), axis=axis)

    filtered_data = np.ma.array(data, copy=copy)

    if iters is None:
        i = -1
        lastrej = filtered_data.count() + 1
        while filtered_data.count() != lastrej:
            i += 1
            lastrej = filtered_data.count()
            do = filtered_data - cenfunc(filtered_data)
            # if nothing left, we dont update the mask array
            if   lastrej    ==  0:
                continue
            else:
                filtered_data.mask |= do * do > varfunc(filtered_data) * sig ** 2
    else:
        for i in range(iters):
            do = filtered_data - cenfunc(filtered_data)
            filtered_data.mask |= do * do > varfunc(filtered_data) * sig ** 2

    return filtered_data


############################################################
#
# Adaptively sigma_clip as a function of magnitude
#
############################################################

def Adpative_sigma_clip(mag1, mag2, sig=3.0, iters=1, cenfunc=np.ma.median, varfunc=np.var):
    """
    This is the function which will do sigma clip on mag1 - mag2 as a function of mag2.
    In this case, we set mag1 = mag_auto from SE and mag2 = mag_true from the input mock.
    Note that the len(mag1) = len(mag2)
    
    Parameters:
        -`mag1`: 1d array. The first magnitude array.
        -`mag2`: 1d array. The second magnitude array.
        -`sig`: float. The multiplicative factor of the sigma clipping.
        -`iters`: int. The iteration. iters = None means performing sigma clipping until it converges.
        -`cenfunc`: function object. The function which is used to calc the center of the distribution.
        -`varfunc`: function object. The function which is used to calc the width of the distribution. Note this is variance not the std.
        
    Returns:
        -`filtered_data1`:  1d array. The filtered array of mag1.
        -`filtered_data2`:  1d array. The filtered array of mag2.
    """

    # obtain the length
    nobjs           =       len(mag1)
    # sanity check
    if   nobjs   !=   len(mag2): raise RuntimeError("len(mag2) != len(mag1) = ", nobjs)
    
    # First derive the dmag = mag1 - mag2. Then we have dmag v.s. mag2.
    # The important part is we express the histogram in
    # the indice of the bins which each value in the dmag belongs.
    # Use ~0.5 mag as the binning step.
    mag_edges     =    np.linspace(mag2.min(), mag2.max(), int( (mag2.max() - mag2.min()) / 0.1 ) + 1)
    mag_bins      =    0.5 * (mag_edges[1:] + mag_edges[:-1])
    mag_steps     =    mag_edges[1:] - mag_edges[:-1]
    # derive dmag
    dmag          =    mag1 - mag2
    # derive hist digitize
    # mag_edges[i-1] < x <= mag_edges[i] (note that we close the right boundary)
    indice_hist   =    np.digitize(mag2, bins = mag_edges, right = True)

    # ---
    # sigma clip on each mag_bins
    # ---
    # declare an array which do not mask anything for mag1 and mag2
    returned_mask =   np.zeros(nobjs, dtype = np.bool)
    # loop all the bins except the indice_hist = 0
    for i in set(indice_hist) - {0}:
        # select the object in this bin
        i_am_in_this_bin    =   np.where(indice_hist == i)[0]
        # if no object, we pass. Or we do sigma clipping
        if   len(i_am_in_this_bin) == 0:
            pass
        else:
            # sigma clipping on dmag[i_am_in_this_bin]
            filtered_data   =   sigma_clip(
                                data = dmag[i_am_in_this_bin],
                                sig  = sig,
                                iters= iters,
                                cenfunc=cenfunc,
                                varfunc=varfunc,
                                axis=None,
                                copy=True)
            # pass the mask aray to returned_mask
            returned_mask[ i_am_in_this_bin ] = filtered_data.mask.copy()

    return returned_mask





if     __name__  ==  "__main__":
    x2  =   np.random.uniform(0.0, 1000.0, 50000)
    y2  =   np.random.uniform(0.0, 1000.0, 50000)
    z2  =   np.random.uniform(15.0, 30.0, 50000)

    x1  =   np.random.normal(loc = x2, scale = 1.0 / 0.26)
    y1  =   np.random.normal(loc = y2, scale = 1.0 / 0.26)
    z1  =   np.random.normal(loc = z2, scale = 0.1)
    z1err=  np.random.uniform(low = 0.085, high = 0.115, size = len(z1))

    import matplotlib.pyplot as pyplt
    '''
    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
    
    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2
    
    # set kdt for coord2
    kdt   = KDT(coords2)

    # ---
    # start from here
    # ---
    idxs2_within_balls   = kdt.query_ball_point(coords1, 3.0 / 0.26)              # find the neighbors within a ball
    n_within_ball        = np.array(map(len, idxs2_within_balls), dtype = np.int) # counts within each ball
    zero_within_ball     = np.where( n_within_ball == 0)[0]                       # find which one does not have neighbors
    nonzero_within_ball  = np.where( n_within_ball >  0)[0]                       # find which one has neighbors
    
    dz_within_ball       = []   # the distance
    idxs2                = []
    
    for i in nonzero_within_ball:
        dz_temp, matched_id_temp   = KDT( np.transpose([ z2[ idxs2_within_balls[i] ] ]) ).query( np.transpose([ z1[i] ]) )
        matched_id_temp            = idxs2_within_balls[i][ matched_id_temp ]
        dz_within_ball.append(dz_temp)
        idxs2.append(matched_id_temp)

    # index for coord1
    idxs1           = np.arange(x1.size)[ nonzero_within_ball ]
    idxs2           = np.array(idxs2, dtype = np.int)
    dz_within_ball  = np.array(dz_within_ball, dtype = np.float)
    '''
    A = carte2d_and_z_match(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2, stol = 1.0 / 0.26, ztol = 3.0 * z1err )








