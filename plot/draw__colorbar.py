import os, sys
import numpy                        as np
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot2D      as gp2


# ========================================================= #
# ===  draw__colorbar.py                                === #
# ========================================================= #

def draw__colorbar( cmap    =None, nlevels=None, config     =None, \
                    position=None, figsize=None, orientation=None, \
                    pngFile =None, nticks_x=None, nticks_y=None ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( config       is None ): config      = lcf.load__config()
    if ( orientation  is None ): orientation = config["clb.orientation"]
    
    if   ( orientation.lower() in [ "h", "horizontal" ] ):
        xAxis    = np.array( [ 0.0, 1.0, 0.0, 1.0 ] )
        yAxis    = np.array( [ 0.0, 0.0, 1.0, 1.0 ] )
        Data     = xAxis
        if ( nticks_x is None ):
            nticks_x = config["clb.x.major.nticks"]
        nticks_y = 1
        if ( position is None ): position = [ 0.10, 0.10, 0.90, 0.90 ]
    elif ( orientation.lower() in [ "v", "vertical"   ] ):
        xAxis    = np.array( [ 0.0, 1.0, 0.0, 1.0 ] )
        yAxis    = np.array( [ 0.0, 0.0, 1.0, 1.0 ] )
        Data     = yAxis
        if ( nticks_y is None ):
            nticks_y = config["clb.y.major.nticks"]
        nticks_x = 1
        if ( position is None ): position = [ 0.10, 0.10, 0.90, 0.90 ]
        
    if ( figsize  is None ): figsize = config["clb.figure.figsize"]
    if ( cmap     is None ): cmap    = config["cmp.colortable"]
    if ( nlevels  is None ): nlevels = config["cmp.level"]["num"]
    if ( pngFile  is None ): pngFile = config["clb.figure.pngFile"]
    levels  = np.linspace( 0.0, 1.0, nlevels )

    
    # ------------------------------------------------- #
    # --- [2] gplot2d                               --- #
    # ------------------------------------------------- #
    config_  = {
        "figure.size"        : figsize,
        "figure.pngFile"     : pngFile, 
        "figure.position"    : position,
        "ax1.x.range"        : { "auto":False, "min": 0.0, "max":1.0, "num":nticks_x },
        "ax1.y.range"        : { "auto":False, "min": 0.0, "max":1.0, "num":nticks_y },
        "ax1.x.minor.nticks" : 1, 
        "ax1.y.minor.nticks" : 1, 
        "ax1.x.label"        : "", 
        "ax1.y.label"        : "", 
        "ax1.x.major.noLabel": True, 
        "ax1.y.major.noLabel": True, 
        "ax1.x.major.off"    : True, 
        "ax1.y.major.off"    : True, 
        "ax1.x.minor.off"    : True, 
        "ax1.y.minor.off"    : True, 
        "ax1.x.minor.nticks" : 1,
        "cmp.level"          : { "auto":False, "min": 0.0, "max":1.0, "num":nlevels },
        "cmp.colortable"     : cmap,
        "clb.sw"             : False,
        "grid.major.sw"      : False,
        "grid.minor.sw"      : False,
    }
    config = { **config, **config_ }
    
    fig    = gp2.gplot2D( xAxis=xAxis, yAxis=yAxis, cMap=Data, config=config )

    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    pngFile     = "../test/draw__colorbar/colorbar.png"
    nlevels     = 50
    orientation = "v"
    figsize     = [1,5]
    nticks_y    = 5
    draw__colorbar( pngFile=pngFile, orientation=orientation, figsize=figsize, \
                    nlevels=nlevels, nticks_y=nticks_y )

