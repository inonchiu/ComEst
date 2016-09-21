#!/usr/bin/env python


##############################################
#
# This module contains some utilities
#
##############################################


class argpasser(object):
    """
    ComEst use the arguments that are almost repeatedly. Therefore, it will be useful to create a customized arguemnt passer like this.

    """
    def __init__(self,
                 stamp_size_arcsec  = 20.0,
                 mag_dict           = {"lo":20.0, "hi":25.0 },
                 hlr_dict           = {"lo":0.35, "hi":0.75 },
                 fbulge_dict        = {"lo":0.5 , "hi":0.9  },
                 q_dict             = {"lo":0.4 , "hi":1.0  },
                 pos_ang_dict       = {"lo":0.0 , "hi":180.0},
                 ngals_arcmin2      = 15.0,
                 nsimimages         = 50,
                 ncpu               = 2,
                 ):
        """
        :param stamp_size_arcsec: The size of the stamp of each simulated source by **GalSim**. The stamp is with the size of ``stamp_size_arcsec`` x ``stamp_size_arcsec`` (``stamp_size_arcsec`` in arcsec) where the **GalSim** will simulate one single source on. By default, it is ``stamp_size_arcsec = 15.0``.

        :param mag_dict: The magnitude range which **GalSim** will simulate sources. It must be in the form of ``{"lo": _value_, "hi": _value_}``, where _value_ is expressed in magnitude. By default, it is ``mag_dict = {"lo":20.0, "hi":25.0 }``.

        :param hlr_dict: The half light radius configuration of the sources simulated by **GalSim**. It is in the unit of arcsec. It has to be in the form of ``{"lo": _value_, "high": _value_}``. By default, it is ``hlr_dict = {"lo":0.35 , "hi":0.75 }``.

        :param fbulge_dict: The configuration of the fraction of the bulge component. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and 1 means the galaxy has zero fraction of light from the disk component. By default, it is ``fbulge_dict = {"lo":0.5 , "hi":0.9  }``.

        :param q_dict: The minor-to-major axis ratio configuration of the sources simulated by **GalSim**. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,1] and ``q = 1`` means spherical. By default, it is ``q_dict = {"lo":0.4 , "hi":1.0 }``.

        :param pos_ang_dict: The position angle configuration of the sources simulated by **GalSim**. It is in the unit of degree. It must be in the form of ``{"lo": _value_, "high": _value_}``. Note that the _value_ has to be within [0,180.0] and it is counter-clockwise with +x is 0 degree. By default, it is ``pos_ang_dict={"lo":0.0 , "hi":180.0 }``.

        :param ngals_arcmin2: The projected number of the sources simulated by **GalSim** per arcmin square. You dont want to set this number too high because it will cause the problem from blending in the source detection. However, you dont want to lose the statistic power if you set this number too low. By defualt, it is ``ngals_arcmin2 = 15.0``.

        :param nsimimages: The number of the images you want to simulate. It will be saved in the multi-extension file with the code name ``sims_nameroot``. By default, it is ``nsimimages = 50``.

        :param ncpu: The number of cpu for parallel running. By default, it is ``ncpu = 2``. Please do not set this number higher than the CPU cores you have.

        """
        self.stamp_size_arcsec  =   float(stamp_size_arcsec)
        self.mag_dict           =   mag_dict
        self.hlr_dict           =   hlr_dict
        self.fbulge_dict        =   fbulge_dict
        self.q_dict             =   q_dict
        self.pos_ang_dict       =   pos_ang_dict
        self.ngals_arcmin2      =   float(ngals_arcmin2)
        self.nsimimages         =   int(nsimimages)
        self.ncpu               =   int(ncpu)

        return

    # i_am function
    def i_am(self):
        """
        """
        print "#", "stamp_size_arcsec:", self.stamp_size_arcsec
        print "#", "mag_dict:", self.mag_dict
        print "#", "hlr_dict:", self.hlr_dict
        print "#", "fbulge_dict:", self.fbulge_dict
        print "#", "q_dict:", self.q_dict
        print "#", "pos_ang_dict:", self.pos_ang_dict
        print "#", "ngals_arcmin2:", self.ngals_arcmin2
        print "#", "nsimimages:", self.nsimimages
        print "#", "ncpu:", self.ncpu

        return
