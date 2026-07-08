import numpy  as np
import pandas as pd
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  plot histgrams                                   === #
# ========================================================= #

def plot__histgrams():

    inpFile1 = "dat/mesh_quality_FrontalHXT.csv"
    inpFile2 = "dat/mesh_quality_MeshAdapt.csv"
    
    df1 = pd.read_csv( inpFile1 )
    df2 = pd.read_csv( inpFile2 )
    
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [5.0,2.5],
        "figure.pngFile"     : "png/mesh_histgram.png", 
        "figure.position"    : [ 0.16, 0.20, 0.94, 0.92 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":False, "min": 0.0, "max": 1.0, "num":6 },
        "ax1.y.range"        : { "auto":False, "min": 0.0, "max":20.0, "num":5 },
        "ax1.x.label"        : r"$\Gamma$",
        "ax1.y.label"        : "Fraction of elements (%)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 1.8,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
    label1 = "Frontal-Delaunay / HXT"
    label2 = "MeshAdapt / Delaunay"
    
    fig        = gp1.gplot1D( config=config )
    hist_norm1 = df1["histgram"] / np.sum( df1["histgram"] ) * 100.0
    hist_norm2 = df2["histgram"] / np.sum( df2["histgram"] ) * 100.0
    fig.add__bar ( xAxis=df1["bin_centers"], yAxis=hist_norm1, label=label1 )
    fig.add__plot( xAxis=df2["bin_centers"], yAxis=hist_norm2, label=label2 )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    plot__histgrams()
