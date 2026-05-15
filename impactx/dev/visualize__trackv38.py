import os, sys, json5, h5py, tqdm, subprocess, pathlib, glob
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import nk_toolkit.plot.load__config       as lcf
import nk_toolkit.plot.gplot1D            as gp1


# ========================================================= #
# ===  visualize__main                                  === #
# ========================================================= #

def visualize__main( stat=False, poincare=False, plot_conf=None, ):

    # ------------------------------------------------- #
    # --- [1] plot statistics                       --- #
    # ------------------------------------------------- #
    if ( stat ):
        df_stat = pd.read_csv( plot_conf["files"]["statistics"], sep=r"\s+" )
        plot__beamStats( df=df_stat, plot_conf=plot_conf )

        fileList = [ "s-xRange", "s-yRange", "s-phiRange", "s-sigma_xy", "s-sigma_phi",
                     "s-alpha_xyz", "s-emit_xyz", "s-emit_xyzn", "s-beta_xy", "s-beta_z", ]
        cmd      = [ "magick", "montage" ] + [ f"png/stat/{afile}.png" for afile in fileList ] \
            + [ "-tile", "2x", "-geometry", "+0+0", "png/stat/stat-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/stat/stat-merged.png" )

        
    # ------------------------------------------------- #
    # --- [2] plot poincare                         --- #
    # ------------------------------------------------- #
    if ( poincare ):
        df_coor = pd.read_csv( plot_conf["files"]["coordinate"], sep=r"\s+" )
        plot__poincareMap( df=df_coor, plot_conf=plot_conf )

        plotNames = [ "xp-px", "yp-py", "dt-dE" ]
        pdir      = pathlib.Path( "png/poincare/" )
        indices   = sorted( [ afile.stem.rsplit("_", 1)[1]
                              for afile in pdir.glob(f"{plotNames[0]}_*.png") ], key=int, )
        fileList  = []
        for idx in indices:
            row        = [ pdir / f"{plotName}_{idx}.png" for plotName in plotNames ]
            missing    = [ str(afile) for afile in row if not afile.exists() ]
            if missing:
                raise FileNotFoundError(f"missing poincare plot(s): {missing}")
            fileList  += [ str(afile) for afile in row ]
        cmd      = [ "magick", "montage" ] + fileList \
            + [ "-tile", "3x", "-geometry", "+0+0", "png/poincare/poincare-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/poincare/poincare-merged.png" )
    
        
    return()




# ========================================================= #
# ===  plot__beamStats                                  === #
# ========================================================= #

def plot__beamStats( df=None, plot_conf=None, kind="stat"):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase  = f"png/{kind}/" + "{}.png"
    basedir  = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )
    
    if ( plot_conf is None ):
        raise ValueError( " [ERROR] plot_conf is None... " )
    if not( kind in plot_conf  ):
        raise KeyError( f" [ERROR] plot_conf does not include {kind} plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf[kind].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf[kind]["default"] }

    # ------------------------------------------------- #
    # --- [2] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        hconf  = contents["config"]
        config = { **def_conf, **hconf }
        fig    = gp1.gplot1D( config=config, pngFile=pngbase.format( key ) )
        for plot in contents["plots"]:
            plot_ax  = { "xAxis":df[plot["xAxis"]], "yAxis":df[plot["yAxis"]] }
            plot_opt = { key:val for key,val in plot.items() if key not in [ "xAxis","yAxis" ] }
            fig.add__plot( **plot_ax, **plot_opt, **contents["option"] )
        if ( len( contents["plots"] ) >= 2 ):
            fig.set__legend()
        fig.set__axis()
        fig.save__figure()

        

# ========================================================= #
# ===  plot__poincareMap                                === #
# ========================================================= #

def plot__poincareMap( df=None, plot_conf=None, step=0 ):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase  = "png/poincare/{{}}_{:06}.png".format( step )
    basedir  = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )
    
    if ( plot_conf is None ):
        raise TypeError( " [ERROR] plot_conf is None... " )
    if not( "poincare" in plot_conf  ):
        raise KeyError( " [ERROR] plot_conf does not include 'poincare' plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf["poincare"].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf["poincare"]["default"] }

    # ------------------------------------------------- #
    # --- [2] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        hconf  = contents["config"]
        config = { **def_conf, **hconf }
        fig    = gp1.gplot1D( config=config, pngFile=pngbase.format( key ) )
        for plot in contents["plots"]:
            plot_ax  = { "xAxis":df[plot["xAxis"]], "yAxis":df[plot["yAxis"]] }
            plot_opt = { key:val for key,val in plot.items() if key not in [ "xAxis","yAxis" ] }
            fig.add__scatter( **plot_ax, **plot_opt, **contents["option"] )
        fig.set__axis()
        fig.save__figure()
        


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    stat     = True
    poincare = True
    
    inpFile = "dat/visualize.json"
    with open( inpFile, "r" ) as f:
        plot_conf = json5.load( f )

    visualize__main( stat=stat, poincare=poincare, plot_conf=plot_conf )
