import os, sys, json5
import numpy  as np
import pandas as pd
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  plot__crossSection.py                            === #
# ========================================================= #

def plot__crossSection( inpFiles=[], pngFile=None, labels=None, JENDL=True ):

    MeV, mb = 1.0e6, 1.0e-3
    E_, xs_ = 0, 1

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( len(inpFiles) <= 0 ):
        raise ValueError( "[plot__crossSection.py] inpFiles :: ".format( inpFiles ) )
    if ( pngFile is None ): pngFile = "xs.png"
    if ( labels  is None ): labels  = inpFiles
    if ( isinstance( JENDL, bool ) ):
        JENDL   = [ JENDL for ik in range( len( inpFiles ) ) ]

    # ------------------------------------------------- #
    # --- [2] fetch data                            --- #
    # ------------------------------------------------- #
    E_xs_list = []
    for ik,inpFile in enumerate(inpFiles):
        with open( inpFile, "r" ) as f:
            E_xs = np.loadtxt( f, comments="#" )
        if ( JENDL[ik] ):
            E_xs[:,E_ ] = E_xs[:,E_ ] / MeV
            E_xs[:,xs_] = E_xs[:,xs_] / mb
        else: 
            pass
        E_xs_list += [ E_xs ]

    # ------------------------------------------------- #
    # --- [2] config                                --- #
    # ------------------------------------------------- #
    config  = {
        **lcf.load__config(), 
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : pngFile, 
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "Energy [MeV]",
        "ax1.y.label"        : "Cross section [mb]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 9.0, 
    }

    # ------------------------------------------------- #
    # --- [3] plot data                             --- #
    # ------------------------------------------------- #
    fig    = gp1.gplot1D( config=config )
    for ik,E_xs in enumerate(E_xs_list):
        fig.add__plot( xAxis=E_xs[:,E_], yAxis=E_xs[:,xs_], label=labels[ik] )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    inpFiles = [ "xs/xs__JENDL5_Ra226gn.dat" ]
    
    plot__crossSection( inpFiles=inpFiles )


