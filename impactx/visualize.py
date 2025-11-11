import sys
import numpy  as np
import pandas as pd
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  plot__poincare.py                                === #
# ========================================================= #

def plot__poincare( paramsFile="dat/parameters.json", steps=None, pids=None ):

    mm, mrad  = 1.0e-3, 1.0e-3
    ns        = 1.0e-9
    cv        = 2.99792458e8
    
    # ------------------------------------------------- #
    # --- [1] get particle info                     --- #
    # ------------------------------------------------- #
    import nk_toolkit.impactx.impactx_toolkit as itk
    ptcls        = itk.get__particles( paramsFile=paramsFile, steps=steps, pids=pids )
    ptcls["xp"]  = ptcls["xp"]  / mm
    ptcls["yp"]  = ptcls["yp"]  / mm
    ptcls["px"]  = ptcls["px"]  / mrad
    ptcls["py"]  = ptcls["py"]  / mrad
    ptcls["dEk"] = ptcls["dEk"]
    ptcls["dt"]  = ptcls["dt"]  / ns
    ptcls["tp"]  = ptcls["tp"]  / mm
    ptcls["pt"]  = ptcls["pt"]  / mrad
    if ( isinstance( steps, (int,float) ) ):
        steps = sorted( list( set( ptcls["step"] ) ) )

        
    # ------------------------------------------------- #
    # --- [2] default config                        --- #
    # ------------------------------------------------- #
    config  = lcf.load__config()
    config_ = {
        "figure.size"        : [4.0,4.0],
        "figure.position"    : [ 0.18, 0.18, 0.92, 0.92 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.minor.nticks" : 1,
        "plot.linestyle"     : "none", 
        "plot.marker"        : "o",
        "plot.markersize"    : 0.2,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [2-1] x-xp config                         --- #
    # ------------------------------------------------- #
    config_  = {
        "ax1.x.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
        "ax1.y.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
        "ax1.x.label"        : "x [mm]",
        "ax1.y.label"        : "x' [mrad]",
    }
    config_x  = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [2-2] y-yp config                         --- #
    # ------------------------------------------------- #
    config_  = {
        "ax1.x.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
        "ax1.y.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
        "ax1.x.label"        : "y [mm]",
        "ax1.y.label"        : "y' [mrad]",
    }
    config_y  = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [2-3] t-tp config                         --- #
    # ------------------------------------------------- #
    config_  = {
        "ax1.x.range"        : { "auto":False, "min": -300.0, "max":300.0, "num":6 },
        "ax1.y.range"        : { "auto":False, "min": -3.000, "max":3.000, "num":6 },
        "ax1.x.label"        : "t [mm]",
        "ax1.y.label"        : "t' [mrad]",
    }
    config_t  = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [2-4] dt-dEk config                       --- #
    # ------------------------------------------------- #
    config_  = {
        "ax1.x.range"        : { "auto":False, "min": -3.000, "max":3.000, "num":11 },
        "ax1.y.range"        : { "auto":False, "min": -0.300, "max":0.300, "num":11 },
        "ax1.x.label"        : "dt [nsec]",
        "ax1.y.label"        : "dW [MeV/u]",
    }
    config_dEk  = { **config, **config_ }

    
    # ------------------------------------------------- #
    # --- [3] plot config                           --- #
    # ------------------------------------------------- #
    for ik in steps:
        
        # ------------------------------------------------- #
        # --- [3-1] x-xp                                --- #
        # ------------------------------------------------- #
        pngFile = "png/poincare_x-xp__{:04}.png".format( ik )
        fig     = gp1.gplot1D( config=config_x, pngFile=pngFile )
        ptcls_  = ptcls[ ptcls["step"] == ik ]
        fig.add__scatter( xAxis=ptcls_["xp"], yAxis=ptcls_["px"], density=True )
        fig.set__axis()
        fig.save__figure()
        
        # ------------------------------------------------- #
        # --- [3-2] y-yp                                --- #
        # ------------------------------------------------- #
        pngFile = "png/poincare_y-yp__{:04}.png".format( ik )
        fig     = gp1.gplot1D( config=config_y, pngFile=pngFile )
        ptcls_  = ptcls[ ptcls["step"] == ik ]
        fig.add__scatter( xAxis=ptcls_["yp"], yAxis=ptcls_["py"], density=True )
        fig.set__axis()
        fig.save__figure()
        
        # ------------------------------------------------- #
        # --- [3-3] t-tp                                --- #
        # ------------------------------------------------- #
        pngFile = "png/poincare_t-tp__{:04}.png".format( ik )
        fig     = gp1.gplot1D( config=config_t, pngFile=pngFile )
        ptcls_  = ptcls[ ptcls["step"] == ik ]
        fig.add__scatter( xAxis=ptcls_["tp"], yAxis=ptcls_["pt"], density=True )
        fig.set__axis()
        fig.save__figure()
        
        # ------------------------------------------------- #
        # --- [3-4] dt-dEk                              --- #
        # ------------------------------------------------- #
        pngFile = "png/poincare_dt-dEk__{:04}.png".format( ik )
        fig     = gp1.gplot1D( config=config_dEk, pngFile=pngFile )
        ptcls_  = ptcls[ ptcls["step"] == ik ]
        fig.add__scatter( xAxis=ptcls_["dEk"], yAxis=ptcls_["dt"], density=True )
        fig.set__axis()
        fig.save__figure()
        
    return()



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    plot__poincare( steps=20 )
