import os, sys, json5
import numpy  as np
import pandas as pd
import nk_toolkit.madx.load__tfs      as ltf
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1

# ========================================================= #
# ===  calculate__energy.py                             === #
# ========================================================= #
def calculate__energy( paramsFile="dat/parameters.json" ):

    MeV2GeV  = 1.e6 / 1.e9
    
    # ------------------------------------------------- #
    # --- [1] load files                            --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    track    = ( ltf.load__tfs( tfsFile=params["post.track.inpFile"] ) )["df"]
    track    = track[ track["NUMBER"]==1 ]

    # ------------------------------------------------- #
    # --- [2] calculate                             --- #
    # ------------------------------------------------- #
    sL       = track["S"].values                           # [m]
    E0ref    = params["umass"] * params["mass/u"]          # [MeV]
    Ekref    = params["init.Ek"]                           # [MeV]
    pcref    = np.sqrt( Ekref**2 + 2.0*Ekref*E0ref )       # [MeV]
    dE       = track["PT"].values * pcref                  # [MeV]
    Ek       = Ekref + dE
    
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : "png/s_Ekinetic.png", 
        "figure.position"    : [ 0.18, 0.18, 0.94, 0.94 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":80.0, "num":9 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":100.0, "num":11 },
        "ax1.x.label"        : r"$s$  (m)",
        "ax1.y.label"        : r"$E_k$ (MeV)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
    
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=sL, yAxis=Ek, label="kinetic [MeV]" )
    fig.set__axis()
    fig.save__figure()

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    calculate__energy()
