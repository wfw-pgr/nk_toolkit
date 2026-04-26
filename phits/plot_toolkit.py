import numpy as np
import os, sys, json5
import numpy  as np
import pandas as pd
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1
import nk_toolkit.phits.io_toolkit    as itk


# ========================================================= #
# ===  plot__fluence_vs_energy.py                       === #
# ========================================================= #
def plot__fluence_vs_energy( inpFile=None, pngFile="fluence_vs_energy.png", \
                             x_lower="e-lower", x_upper="e-upper", y_label="neutron", \
                             config={} ):
    
    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    if not( isinstance( inpFile, str ) ):
        raise TypeError ( "[ERROR] inpFile must be str" )
    if not( os.path.exists( inpFile )  ):
        raise FileNotFoundError( "[ERROR] inpFile not found :: {}".format( inpFile ) )
    dfList = itk.read__phitsANGEL( inpFile=inpFile )

    # ------------------------------------------------- #
    # --- [2] column_stack & ravel for step plot    --- #
    # ------------------------------------------------- #
    stack = []
    for df in dfList:
        xAxis  = np.column_stack( [ df[x_lower].to_numpy(), df[x_upper].to_numpy() ] ).ravel()
        yAxis  = np.column_stack( [ df[y_label].to_numpy(), df[y_label].to_numpy() ] ).ravel()
        stack += [ pd.DataFrame.from_dict( { "xAxis":xAxis, "yAxis":yAxis } ) ]
    
    # ------------------------------------------------- #
    # --- [2] config                                --- #
    # ------------------------------------------------- #
    config  = {
        **lcf.load__config(), 
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : pngFile, 
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "Energy (MeV)",
        "ax1.y.label"        : "Track length [m/MeV/s]",
        "ax1.x.minor.nticks" : 1, 
        "ax1.x.log"          : False,  
        "ax1.y.log"          : False,
        "plot.marker"        : "none",
        "plot.linestyle"     : "-",
        "legend.fontsize"    : 9.0,
        **config, 
    }
    
    # ------------------------------------------------- #
    # --- [3] plot data                             --- #
    # ------------------------------------------------- #
    fig    = gp1.gplot1D( config=config )
    for df in stack:
        fig.add__plot( xAxis=df["xAxis"], yAxis=df["yAxis"] )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    config  = {
        "ax1.x.log"   :True, \
        "ax1.y.log"   :True, \
        "ax1.x.range" : { "auto":False, "min": 1.0e-10, "max":1.0e2, "num":13 },
        "ax1.y.range" : { "auto":False, "min": 1.0e+6,  "max":1.0e14, "num":9 },
    }
    inpFile = "out/fluence_n_energy.dat"
    plot__fluence_vs_energy( inpFile=inpFile, config=config )
    
