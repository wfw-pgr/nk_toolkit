import os, sys
import numpy                        as np
import pandas                       as pd
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1
import nk_toolkit.opal.opal_toolkit as opk


# ========================================================= #
# ===  plot__statistics.py                              === #
# ========================================================= #
def plot__statistics( inpFile=None, keys=None ):

    soloKeys = [ "t", "s", "numParticles", "charge", "energy", "dE", "dt", \
                 "partsOutside", "DebyeLength", "plasmaParameter", "temperature", "rmsDensity" ]
    xysKeys  = [ "rms_", "rms_p", "emit_", "mean_", "max_",  ]
    xyzKeys  = [ "ref_", "ref_p" ]
    ylabels_ = { "t":"t [ns]", "s":"s [m]", "charge":"charge [C]", "energy":"energy [MeV]", "dE":"Energy Spread [MeV]", \
                 "rms_":"RMS beam size [m]", "rms_p":"RMS momentum", "emit_":"normalized emittance", \
                 "mean_":"mean position [m]", "max_":"max beam size [m]", "ref_":"reference position [m]", "ref_p":"reference momentum" ,\
                 "B": "B [T]", "E":"E [MV/m]", "D":"Dispersion [m]", "DD":"Derivatives of Dispersion", "correlation": "correlation" }
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data     = opk.load__statistics( inpFile=inpFile )
    if ( keys is None ): keys = data.keys()
    ylabels  = { key:key for key in keys }
    ylabels  = { **ylabels, **ylabels_ }
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] plot each graph                       --- #
    # ------------------------------------------------- #
    for key in soloKeys:
        config_  = {
            "figure.pngFile"     : "test/statistics__{}.png".format( key ), 
            "ax1.y.label"        : ylabels[key],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[key] )
        fig.set__axis()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] xyz / xys variables                   --- #
    # ------------------------------------------------- #
    for key in xysKeys:
        config_  = {
            "figure.pngFile"     : "test/statistics__{}xys.png".format( key ), 
            "ax1.y.label"        : ylabels[key],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"x"], label=key+"x" )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"y"], label=key+"y" )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"s"], label=key+"s" )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()
        
    for key in xyzKeys:
        config_  = {
            "figure.pngFile"     : "test/statistics__{}xyz.png".format( key ), 
            "ax1.y.label"        : ylabels[key],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"x"], label=key+"x" )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"y"], label=key+"y" )
        fig.add__plot( xAxis=data["s"], yAxis=data[key+"z"], label=key+"z" )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] BField                                --- #
    # ------------------------------------------------- #
    key      = "B"
    config_  = {
        "figure.pngFile"     : "test/statistics__B.png", 
        "ax1.y.label"        : ylabels[key],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["Bx_ref"], label=key+"x" )
    fig.add__plot( xAxis=data["s"], yAxis=data["By_ref"], label=key+"y" )
    fig.add__plot( xAxis=data["s"], yAxis=data["Bz_ref"], label=key+"z" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()
    
    # ------------------------------------------------- #
    # --- [6] EField                                --- #
    # ------------------------------------------------- #
    key      = "E"
    config_  = {
        "figure.pngFile"     : "test/statistics__E.png", 
        "ax1.y.label"        : ylabels[key],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["Ex_ref"], label=key+"x" )
    fig.add__plot( xAxis=data["s"], yAxis=data["Ey_ref"], label=key+"y" )
    fig.add__plot( xAxis=data["s"], yAxis=data["Ez_ref"], label=key+"z" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [7] Dispersion                            --- #
    # ------------------------------------------------- #
    key      = "D"
    config_  = {
        "figure.pngFile"     : "test/statistics__dispersion.png", 
        "ax1.y.label"        : ylabels[key],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["Dx"], label=key+"x" )
    fig.add__plot( xAxis=data["s"], yAxis=data["Dy"], label=key+"y" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    key      = "DD"
    config_  = {
        "figure.pngFile"     : "test/statistics__dispersion_deriv.png", 
        "ax1.y.label"        : ylabels[key],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["DDx"], label=key+"x" )
    fig.add__plot( xAxis=data["s"], yAxis=data["DDy"], label=key+"y" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [8] correlation                           --- #
    # ------------------------------------------------- #
    key      = "correlation"
    config_  = {
        "figure.pngFile"     : "test/statistics__correlation.png", 
        "ax1.y.label"        : ylabels[key],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["xpx"], label="xpx" )
    fig.add__plot( xAxis=data["s"], yAxis=data["ypy"], label="ypy" )
    fig.add__plot( xAxis=data["s"], yAxis=data["zpz"], label="zpz" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "test/main.stat"
    plot__statistics( inpFile=inpFile )
