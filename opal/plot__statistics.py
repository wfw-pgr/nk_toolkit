import os, sys
import numpy                        as np
import pandas                       as pd
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1
import nk_toolkit.opal.opal_toolkit as opk


# ========================================================= #
# ===  plot__statistics.py                                === #
# ========================================================= #
def plot__statistics( inpFile=None, keys=None ):

    ylabels = {}
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data = opk.load__sdds( inpFile=inpFile )
    if ( keys is None ): keys = data.keys()
    ylabels = { key:key for key in keys }
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [9.0,4.5],
        "figure.position"    : [ 0.08, 0.14, 0.92, 0.92 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] plot each graph                       --- #
    # ------------------------------------------------- #
    for key in keys:
        config_  = {
            "figure.pngFile"     : "png/statistics__{}.png".format( key ), 
            "ax1.y.label"        : ylabels[key],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[key], label="sample" )
        fig.set__axis()
        fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "sample.stat"
    plot__statistics( inpFile=inpFile )
