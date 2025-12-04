import os, sys
import numpy as np
import pandas as pd
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  plot__initialDist                                === #
# ========================================================= #

def plot__initialDist( trackFile  ="track/initial_coord.out", \
                       impactxFile="impactx/diags/openPMD/bpm.h5", \
                       Ek0=40.0, Em0=2.014*931.494, Nu=2.0  ):
    initialDist__track  ( inpFile=trackFile  , Ek0=Ek0, Em0=Em0, Nu=Nu )
    initialDist__impactx( inpFile=impactxFile, Ek0=Ek0, Em0=Em0, Nu=Nu )


# ========================================================= #
# ===  set__config.py                                   === #
# ========================================================= #

def set__config( target=None ):

    # ------------------------------------------------- #
    # --- [1] default config settings               --- #
    # ------------------------------------------------- #
    config  = lcf.load__config()
    config_ = {
        "figure.size"        : [4.5,4.5],
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
    # --- [2] config setting by target              --- #
    # ------------------------------------------------- #
    if   ( target == "x" ):
        config_  = {
            "ax1.x.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
            "ax1.x.label"        : "x [mm]",
            "ax1.y.label"        : "x' [mrad]",
        }
    elif ( target == "y" ):
        config_  = {
            "ax1.x.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -100.0, "max":100.0, "num":11 },
            "ax1.x.label"        : "y [mm]",
            "ax1.y.label"        : "y' [mrad]",
        }
    elif ( target == "w" ):
        config_  = {
            "ax1.x.range"        : { "auto":False, "min": -1.000, "max":1.000, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -0.100, "max":0.100, "num":11 },
            "ax1.x.label"        : "dt [nsec]",
            "ax1.y.label"        : "dW [MeV/u]",
        }
    elif ( target == "t" ):
        config_  = {
            "ax1.x.range"        : { "auto":False, "min": -300.0, "max":300.0, "num":6 },
            "ax1.y.range"        : { "auto":False, "min": -3.000, "max":3.000, "num":6 },
            "ax1.x.label"        : "t [mm]",
            "ax1.y.label"        : "t' [mrad]",
        }
    else:
        sys.exit( "unknown target :: {}".format( target ) )
    config = { **config, **config_ }
    return( config )


# ========================================================= #
# ===  initialDist__impactx.py                          === #
# ========================================================= #

def initialDist__impactx( inpFile="impactx/diags/openPMD/bpm.h5", \
                          outDir ="png/initialDist/", \
                          Ek0=40.0, Em0=2.014*931.494, Nu=2.0 ):

    mm, mrad  = 1.0e-3, 1.0e-3
    cv, nsec  = 2.99792458e8, 1.0e-9
    os.makedirs( outDir, exist_ok=True )
    
    # ------------------------------------------------- #
    # --- [2] load particle's data                  --- #
    # ------------------------------------------------- #
    import nk_toolkit.impactx.impactx_toolkit as itk
    ret       = itk.load__impactHDF5( inpFile=inpFile, steps=[1] )
    ret["xp"] = ret["xp"] / mm
    ret["yp"] = ret["yp"] / mm
    ret["tp"] = ret["tp"] / mm
    ret["px"] = ret["px"] / mrad
    ret["py"] = ret["py"] / mrad
    ret["pt"] = ret["pt"] / mrad

    Etot      = Em0 + Ek0
    p0c       = np.sqrt( Etot**2 - Em0**2 )
    ret["dt"] = ret["tp"] * mm / cv / nsec
    ret["dW"] = ret["pt"] * mrad * p0c / Nu     # Nu :: #.of nucleon   dW = [MeV/u]


    # ------------------------------------------------- #
    # --- [3] x-xp                                  --- #
    # ------------------------------------------------- #
    config = set__config( target="x" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"x-xp_impactx.png" )
    fig.add__scatter( xAxis=ret["xp"], yAxis=ret["px"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] y-yp                                  --- #
    # ------------------------------------------------- #
    config = set__config( target="y" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"y-yp_impactx.png" )
    fig.add__scatter( xAxis=ret["yp"], yAxis=ret["py"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] t-tp                                  --- #
    # ------------------------------------------------- #
    config = set__config( target="t" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"t-tp_impactx.png" )
    fig.add__scatter( xAxis=ret["tp"], yAxis=ret["pt"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [6] dt-dW                                 --- #
    # ------------------------------------------------- #
    config = set__config( target="w" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"dt-dW_impactx.png" )
    fig.add__scatter( xAxis=ret["dt"], yAxis=ret["dW"], density=True )
    fig.set__axis()
    fig.save__figure()


# ========================================================= #
# ===  initial__track.py                                === #
# ========================================================= #

def initialDist__track( inpFile="track/initial_coord.out", \
                        outDir ="png/initialDist/", \
                        Ek0=40.0, Em0=2.014*931.494, Nu=2.0 ):
    
    cm2mm     = 10.0
    mm, mrad  = 1.0e-3, 1.0e-3
    cv, nsec  = 2.99792458e8, 1.0e-9
    os.makedirs( outDir, exist_ok=True )

    # ------------------------------------------------- #
    # --- [1] load track's data                     --- #
    # ------------------------------------------------- #
    Etot              = Em0 + Ek0
    p0c               = np.sqrt( Etot**2 - Em0**2 )
    track             = pd.read_csv( inpFile, sep=r"\s+" )
    track["x[mm]"]    = track["x[cm]"] * cm2mm
    track["y[mm]"]    = track["y[cm]"] * cm2mm
    track["t[mm]"]    = track["dt[nsec]"] * nsec * cv / mm
    track["t'[mrad]"] = track["dW[Mev/u]"] * Nu   / p0c / mrad

    # ------------------------------------------------- #
    # --- [2] x-xp                                  --- #
    # ------------------------------------------------- #
    config = set__config( target="x" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"x-xp_Track.png" )
    fig.add__scatter( xAxis=track["x[mm]"], yAxis=track["x'[mrad]"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [3] y-yp                                  --- #
    # ------------------------------------------------- #
    config = set__config( target="y" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"y-yp_Track.png" )
    fig.add__scatter( xAxis=track["y[mm]"], yAxis=track["y'[mrad]"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] dt-dW                                 --- #
    # ------------------------------------------------- #
    config = set__config( target="w" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"dt-dW_Track.png" )
    fig.add__scatter( xAxis=track["dt[nsec]"], yAxis=track["dW[Mev/u]"], density=True )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] t-tp ( t=cdt - tp=p/p0=g/(g+1)dW/W )  --- #
    # ------------------------------------------------- #
    config = set__config( target="t" )
    fig    = gp1.gplot1D( config=config, pngFile=outDir+"t-tp_Track.png" )
    fig.add__scatter( xAxis=track["t[mm]"], yAxis=track["t'[mrad]"], density=True )
    fig.set__axis()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    plot__initialDist( trackFile  ="track/initial_coord.out", \
                       impactxFile="impactx/diags/openPMD/bpm.h5", \
                       Ek0=40.0, Em0=2.014*931.494, Nu=2.0 )
