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
# ===  compare__impactx_vs_trackv38.py                  === #
# ========================================================= #

def compare__impactx_vs_trackv38( trackFile  ="dat/beam.dat", \
                                  impactxFile="dat/reduced_beam_characteristics.0", \
                                  plot_conf  ={} ):

    cm, mrad = 1.0e-2, 1.0e-3
    
    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    df_trackv38 = pd.read_csv(   trackFile, sep=r"\s+" )
    df_impactx  = pd.read_csv( impactxFile, sep=r"\s+" )
    
    # ------------------------------------------------- #
    # --- [2] make comparable dataframe             --- #
    # ------------------------------------------------- #
    tr = pd.DataFrame()
    im = pd.DataFrame()

    # -- independent variable
    tr["s"] = df_trackv38["dist[m]"]
    im["s"] = df_impactx ["s"]

    # -- rms beam size: TRACK [cm] -> [m]
    tr["sigma_x"] = df_trackv38["x_rms[cm]"] * cm
    tr["sigma_y"] = df_trackv38["y_rms[cm]"] * cm
    im["sigma_x"] = df_impactx["sigma_x"]
    im["sigma_y"] = df_impactx["sigma_y"]

    # -- Twiss beta: TRACK [cm/mrad] -> [m/rad]
    tr["beta_x"] = df_trackv38["b_x[cm/mrad]"] * cm / mrad
    tr["beta_y"] = df_trackv38["b_y[cm/mrad]"] * cm / mrad
    im["beta_x"] = df_impactx["beta_x"]
    im["beta_y"] = df_impactx["beta_y"]

    # -- Twiss alpha: dimensionless
    tr["alpha_x"] = df_trackv38["a_x"]
    tr["alpha_y"] = df_trackv38["a_y"]
    im["alpha_x"] = df_impactx["alpha_x"]
    im["alpha_y"] = df_impactx["alpha_y"]

    # -- normalized rms emittance:
    #    TRACK: 4*epsilon_n,rms [cm*mrad] -> epsilon_n,rms [m*rad]
    tr["emit_xn"] = df_trackv38["4*exn_rms[cm*mrad]"] * cm * mrad / 4.0
    tr["emit_yn"] = df_trackv38["4*eyn_rms[cm*mrad]"] * cm * mrad / 4.0
    im["emit_xn"] = df_impactx["emittance_xn"]
    im["emit_yn"] = df_impactx["emittance_yn"]

    # -- centroid: TRACK [cm] -> [m]
    tr["mean_x"] = df_trackv38["Xc[cm]"] * cm
    tr["mean_y"] = df_trackv38["Yc[cm]"] * cm
    im["mean_x"] = df_impactx["mean_x"]
    im["mean_y"] = df_impactx["mean_y"]

    # -- max envelope: TRACK [cm] -> [m]
    tr["xmax"] = df_trackv38["Xmax[cm]"] * cm
    tr["ymax"] = df_trackv38["Ymax[cm]"] * cm
    im["xmax"] = np.maximum( np.abs( df_impactx["min_x"] - df_impactx["mean_x"] ), \
                             np.abs( df_impactx["max_x"] - df_impactx["mean_x"] ) )
    im["ymax"] = np.maximum( np.abs( df_impactx["min_y"] - df_impactx["mean_y"] ), \
                             np.abs( df_impactx["max_y"] - df_impactx["mean_y"] ) )

    # ------------------------------------------------- #
    # --- longitudinal coordinate conversion        --- #
    # ------------------------------------------------- #
    clight    = 2.99792458e8
    freqb_Hz  = plot_conf["compare"]["settings"]["beam.freq.Hz"]
    m_per_deg = clight / ( 360.0 * freqb_Hz )      # ct = c * phi / (360*f)

    tr["mean_t"]  = df_trackv38["Zc[deg]"]       * m_per_deg
    tr["sigma_t"] = df_trackv38["phi_rms[deg]"]  * m_per_deg
    tr["tmax"]    = df_trackv38["phi_max[deg]"]  * m_per_deg
    im["mean_t"]  = df_impactx["mean_t"]
    im["sigma_t"] = df_impactx["sigma_t"]
    im["tmax"]    = np.maximum( np.abs( df_impactx["min_t"] - df_impactx["mean_t"] ), \
                                np.abs( df_impactx["max_t"] - df_impactx["mean_t"] ) )

    # ------------------------------------------------- #
    # --- longitudinal momentum / energy envelope   --- #
    # ------------------------------------------------- #
    # TRACK DW/W[rel.u.] is max energy envelope, not RMS.
    # Convert kinetic-energy relative spread dW/W to approximate dp/p.
    gamma          = np.sqrt( 1.0 + ( df_trackv38["btgm"] )**2 )
    beta           = df_trackv38["btgm"] / gamma
    dpp_per_dWW    = ( gamma - 1.0 ) / ( beta**2 * gamma )
    tr["ptmax"]    = dpp_per_dWW * df_trackv38["DW/W[rel.u.]"]
    im["ptmax"]    = np.maximum( np.abs( df_impactx["min_pt"] - df_impactx["mean_pt"] ), \
                                 np.abs( df_impactx["max_pt"] - df_impactx["mean_pt"] ) )
    im["sigma_pt"] = df_impactx["sigma_pt"]
    tr["sigma_dW_W"] = 0.01 * df_trackv38["phi_rms[deg]"] / df_trackv38["b_z[deg/(%ofD_W/W)]"] * np.sqrt( 1.0 + df_trackv38["a_z"]**2 )
    tr["sigma_pt"] = gamma / ( gamma + 1.0 ) * tr["sigma_dW_W"]
    im["sigma_pt"] = df_impactx["sigma_pt"]


    # ------------------------------------------------- #
    # --- sanity check  at begining                 --- #
    # ------------------------------------------------- #
    i   = 0
    fac = clight / ( 360.0*freqb_Hz )
    print( "[ t-direction sanity check at begining ]" )
    print( "TRACK phi_rms[deg]  =", df_trackv38["phi_rms[deg]"].iloc[i]     )
    print( "TRACK sigma_t [m]   =", df_trackv38["phi_rms[deg]"].iloc[i]*fac )
    print( "ImpactX sigma_t [m] =", df_impactx["sigma_t"].iloc[i]           )
    print( "ratio TRACK/ImpactX =", df_trackv38["phi_rms[deg]"].iloc[i] \
           * fac/df_impactx["sigma_t"].iloc[i] )
    print()
    print( "TRACK phi_max[deg]  =", df_trackv38["phi_max[deg]"].iloc[i] )
    print( "TRACK tmax [m]      =", df_trackv38["phi_max[deg]"].iloc[i] * fac )
    im_tmax0 = max( abs( df_impactx["min_t"].iloc[i] - df_impactx["mean_t"].iloc[i] ),
                    abs( df_impactx["max_t"].iloc[i] - df_impactx["mean_t"].iloc[i] ), )
    print( "ImpactX tmax [m]    =", im_tmax0 )
    print( "ratio TRACK/ImpactX =", df_trackv38["phi_max[deg]"].iloc[i] * fac / im_tmax0 )
    print()
    ## -- end -- #
    
    df = pd.concat( [ tr.add_prefix( "tr__" ), im.add_prefix( "im__" ) ], axis=1 )

    # ------------------------------------------------- #
    # --- [3] plot settings                         --- #
    # ------------------------------------------------- #
    pngbase  = "png/compare/" + "{}.png"
    basedir  = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )

    if ( plot_conf is None ):
        raise TypeError( " [ERROR] plot_conf is None... " )
    if not( "compare" in plot_conf  ):
        raise KeyError( " [ERROR] plot_conf does not include 'poincare' plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf["compare"].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf["compare"]["default"] }

    # plots_   = {        
    #     "s-sigma_x": {
    #         "config": { "y.label": "$\\sigma_x$ [m]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__sigma_x", "label":"TRACKv38" , "marker":"o", },
    #             { "xAxis":"im__s", "yAxis":"im__sigma_x", "label":"ImpactX"  , "marker":"s", },
    #         ],
    #     },

    #     "s-sigma_y": {
    #         "config": { "y.label":"$\\sigma_y$ [m]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__sigma_y", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__sigma_y", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-beta_x": {
    #         "config": { "y.label": "$\\beta_x$ [m]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__beta_x", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__beta_x", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-beta_y": {
    #         "config": { "y.label": "$\\beta_y$ [m]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__beta_y", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__beta_y", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-alpha_x": {
    #         "config": { "y.label": "$\\alpha_x$" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__alpha_x", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__alpha_x", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-alpha_y": {
    #         "config": { "y.label": "$\\alpha_y$" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__alpha_y", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__alpha_y", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-emit_xn": {
    #         "config": { "y.label": "$\\varepsilon_{x,n}$ [m rad]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__emit_xn", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__emit_xn", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    #     "s-emit_yn": {
    #         "config": { "y.label": "$\\varepsilon_{y,n}$ [m rad]" },
    #         "option": {},
    #         "plots" : [
    #             { "xAxis":"tr__s", "yAxis":"tr__emit_yn", "label":"TRACKv38" , "marker":"o",  },
    #             { "xAxis":"im__s", "yAxis":"im__emit_yn", "label":"ImpactX"  , "marker":"s",  },
    #         ],
    #     },

    # }

    
    # ------------------------------------------------- #
    # --- [4] plot                                  --- #
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

    # ------------------------------------------------- #
    # --- [5] merge files                           --- #
    # ------------------------------------------------- #
    fileList = [ "s-sigma_x", "s-sigma_y", "s-beta_x", "s-beta_y",
                 "s-alpha_x", "s-alpha_y", "s-emit_xn", "s-emit_yn" ,
                 "s-sigma_t", "s-sigma_pt", "s-tmax", "s-mean_t",  ]
    cmd      = [ "magick", "montage" ] + [ f"png/compare/{afile}.png" for afile in fileList ] \
        + [ "-tile", "2x", "-geometry", "+0+0", "png/compare/compare-merged.png" ]
    subprocess.run( cmd, check=True )
    print( "[ magick append @tasks.py ] output :: png/compare/compare-merged.png" )


        

# ========================================================= #
# ===  plot invoke file                                 === #
# ========================================================= #

'''

import json5
import invoke
import nk_toolkit.impactx.trackv38_toolkit as ttk


# ========================================================= #
# ===  plot                                             === #
# ========================================================= #
@invoke.task
def plot( ctx, \
          stat=False, poincare=False, all=False, ext=None, pcnfFile="dat/visualize.json" ):
    """Run plot script."""
    
    with open( pcnfFile, "r" ) as f:
        plot_conf = json5.load( f )
    ttk.visualize__main( stat=stat, poincare=poincare, plot_conf=plot_conf )

'''

