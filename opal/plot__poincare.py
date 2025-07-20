import scipy  as sp
import numpy  as np
import pandas as pd
import matplotlib.colors            as clr
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1
import nk_toolkit.opal.opal_toolkit as opk


# ========================================================= #
# ===  plot__poincare.py                                === #
# ========================================================= #
def plot__poincare( inpFile=None, steps=None ):
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    Data = opk.load__opalHDF5( inpFile=inpFile, steps=steps )

    # ------------------------------------------------- #
    # --- [2] configuration                         --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.20, 0.20, 0.92, 0.92 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":5 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":5 },
        "ax1.x.label"        : "x [m]",
        "ax1.y.label"        : "px [rad]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.linestyle"     : "none", 
        "plot.markersize"    : 0.3,
        "legend.fontsize"    : 9.0, 
    }
    config_x = { **config, **config_ }
    config_y = { "ax1.x.label": "y [m]",
                 "ax1.y.label": "py [rad]" }
    config_y = { **config_x, **config_y }

    for step in steps:
        pngFile = "test/poincare_x__step{}.png".format( step )
        key     = "Step#{}".format( step )
        fig     = gp1.gplot1D( config=config_x, pngFile=pngFile )
        fig.add__scatter( xAxis=Data[key]["x"], yAxis=Data[key]["px"], density=True )
        fig.set__axis()
        fig.save__figure()
        
        pngFile = "test/poincare_y__step{}.png".format( step )
        key     = "Step#{}".format( step )
        fig     = gp1.gplot1D( config=config_y, pngFile=pngFile )
        fig.add__scatter( xAxis=Data[key]["y"], yAxis=Data[key]["py"], density=True )
        fig.set__axis()
        fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "test/main.h5"
    steps   = [ 0 ]
    plot__poincare( inpFile=inpFile, steps=steps )
