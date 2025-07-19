import os, sys
import scipy  as sp
import numpy  as np
import pandas as pd
import matplotlib.colors            as clr
import nk_toolkit.opal.opal_toolkit as opk
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1

# ========================================================= #
# ===  plot__trajectories.py                            === #
# ========================================================= #

def plot__trajectories( hdf5File=None, statFile=None, series=None, obj="x" ):
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    pinfo    = opk.load__opalHDF5  ( inpFile=hdf5File, series=series )
    sinfo    = opk.load__statistics( inpFile=statFile )
    sL       = sinfo["s"]
    nSteps   = len( sL )
    
    # ------------------------------------------------- #
    # --- [2] configuration                         --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.y.label"        : "x [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.linestyle"     : "-", 
        "plot.markersize"    : 1.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    fig    = gp1.gplot1D( config=config )
    for pid in series:
        trajectory = ( pinfo[ pinfo["id"] == pid ] )[obj].values
        if ( len(trajectory) == 0 ): continue
        xAxis  = sL[:(len(trajectory))]
        fig.add__plot( xAxis=xAxis, yAxis=trajectory )
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    npart    = int(1e4)
    nplot    = int(1e4)
    obj      = "x"
    hdf5File = "opal/main.h5"
    statFile = "opal/main.stat"
    series   = np.random.choice( np.arange(1,npart+1), size=nplot, replace=False )
    plot__trajectories( hdf5File=hdf5File, statFile=statFile, series=series, obj=obj )
