import os, sys, json5, h5py, tqdm
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import nk_toolkit.plot.load__config       as lcf
import nk_toolkit.plot.gplot1D            as gp1
import nk_toolkit.impactx.io_toolkit      as itk
import nk_toolkit.impactx.analyze_toolkit as atk


# ========================================================= #
# ===  visualize__main                                  === #
# ========================================================= #

def visualize__main( refp=False, stat=False, poincare=False, post=False, \
                     trajectory=False, plot_conf=None ):
    
    # ------------------------------------------------- #
    # --- [1] main routine                          --- #
    # ------------------------------------------------- #
    files    = plot_conf["files"]

    # ------------------------------------------------- #
    # --- [2] ref_particle plot                     --- #
    # ------------------------------------------------- #
    if ( refp ):
        refp_df = itk.get__beamStats( statFile=files["stat"], refpFile=files["refp"] )
        plot__refparticle( df=refp_df, plot_conf=plot_conf )

    # ------------------------------------------------- #
    # --- [3] reduced beam characteristics plot     --- #
    # ------------------------------------------------- #
    if ( stat ):
        stat_df = itk.get__beamStats( statFile=files["stat"], refpFile=files["refp"] )
        plot__statistics( df=stat_df, plot_conf=plot_conf )

    # ------------------------------------------------- #
    # --- [4] post analysis plot                    --- #
    # ------------------------------------------------- #
    if ( post ):
        df_post = atk.get__postprocessed ( refpFile=files["refp"]  , statFile=files["stat"], \
                                           recoFile=files["record"], postFile=files["post"] )
        plot__postprocessed( df=df_post, plot_conf=plot_conf )


    # ------------------------------------------------- #
    # --- [5] poincare plot                         --- #
    # ------------------------------------------------- #
    if ( poincare ):
        labels    = pd.read_csv( files["label"] )
        steps     = ( labels[ labels["name"] == "bpm" ]["id"] ).to_numpy()
        nPoincare = ( plot_conf.get( "poincare", {} ).get( "settings", {} ) ).get( "nPoincare", 0 )
        if ( nPoincare > 0 ):
            steps   = steps[ np.linspace( 0, len(steps)-1, nPoincare ).astype(int) ]
        for step in steps:
            bpms_df = itk.get__particles( recoFile=files["record"], refpFile=files["refp"], \
                                          bpmsFile=files["bpms"]  , steps   =[step] )
            plot__poincareMap( df=bpms_df, step=step, plot_conf=plot_conf )
            
    # ------------------------------------------------- #
    # --- [6] trajectory plot                       --- #
    # ------------------------------------------------- #
    if ( trajectory ):
        plot__trajectories( plot_conf=plot_conf )

    return()

# ========================================================= #
# ===  plot__lattice                                    === #
# ========================================================= #

def plot__lattice( latticeFile=None, ax=None, \
                   height=1.0, y0=0.0, pngFile=None, label=False ):

    qf_plot = { "color":"royalblue", "alpha":0.8 }
    qd_plot = { "color":"tomato"   , "alpha":0.8 }
    rf_plot = { "color":"orange"   , "alpha":0.8 }
    dr_plot = { "color":"grey"     , "alpha":0.8 }
    
    # ------------------------------------------------- #
    # --- [1] load lattice file                     --- #
    # ------------------------------------------------- #
    with open( latticeFile, "r" ) as f:
        elements = json5.load( f )
    lattice = pd.DataFrame.from_dict( elements, orient="index" )
    lattice = lattice[ [ "name", "type", "ds", "k" ] ]
    ds      = lattice.loc[ :,"ds" ].to_numpy()
    lattice["s_in"]  = np.cumsum( np.insert( ds, 0, 0.0 ) )[:-1]
    lattice["s_out"] = np.cumsum( ds )
    
    # ------------------------------------------------- #
    # --- [2] prepare figure / axis                 --- #
    # ------------------------------------------------- #
    if ( ax is None ):
        fig,ax   = plt.subplots( figsize=(8,2) )
        given_ax = False
    else:
        fig      = ax.figure
        given_ax = True

    # ------------------------------------------------- #
    # --- [3] draw lattice elements                 --- #
    # ------------------------------------------------- #
    for _, row in lattice.iterrows():
        s0,s1  = row["s_in"], row["s_out"]
        etype  = str( row["type"] ).lower()
        name   = str( row["name"] )
        width  = s1 - s0
        k      = row["k"]
        
        # -- [3-1] QF/QD -- #
        if   ( etype in ["quadrupole", "quadrupole.linear" ] ):
            if ( row["k"] > 0 ):
                ax.add_patch( patches.Rectangle( (s0,y0), width, height, **qf_plot ) )
            else:
                ax.add_patch( patches.Rectangle( (s0,y0-height), width, height, **qd_plot ) )
        # -- [3-2] RFcavity -- #
        elif ( etype in ["rfcavity", "rfgap" ] ):
            ax.add_patch( patches.Polygon( [ [s0,y0], [s1,y0+0.5*height], [s1,y0-0.5*height] ],
                                           closed=True, **rf_plot ) )
        # -- [3-3] drift  -- #
        elif ( etype in ["drift","drift.linear"] ):
            ax.plot( [s0,s1], [y0,y0], **dr_plot )
        # -- [3-4] その他（黒線） -- #
        else:
            ax.plot([s0, s1], [y0, y0], color="black", lw=1 )
            
        # --- ラベルを中央に配置 --- #
        if ( label ):
            ax.text( (s0+s1)/2, y0+1.1*height*np.sign(y0+0.1),
                     name, ha="center", va="bottom", fontsize=7 )

    # ------------------------------------------------- #
    # --- [4] axis settings                         --- #
    # ------------------------------------------------- #
    if ( not( given_ax ) ):
        ax.set_xlim( lattice["s_in"].min(), lattice["s_out"].max() )
        ax.set_ylim( -1.5*height, 1.5*height )
        ax.set_xlabel("s [m]")
        ax.set_yticks([])
        ax.grid( False )

        legend_handles = [
            patches.Patch( color="royalblue", label="QF"),
            patches.Patch( color="tomato"   , label="QD"),
            patches.Patch( color="orange"   , label="RFcavity"),
            plt.Line2D([0], [0], color="gray", lw=2, label="Drift")
        ]
        plt.tight_layout()
        ax.legend( handles=legend_handles, loc="upper right", ncol=4, fontsize=6 )
        
        if ( pngFile is not None ):
            ax.axis( "off" )
            ax.set_facecolor("none")
            plt.savefig( pngFile, dpi=300 )
            plt.close()
        else:
            plt.show()

    return( lattice )




# ========================================================= #
# ===  plot__refparticle                                === #
# ========================================================= #

def plot__refparticle( df=None, plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase  = "png/refp/{}.png"
    basedir  = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )
    
    if ( plot_conf is None ):
        raise TypeError( " [ERROR] plot_conf is None... " )
    if not( "refp" in plot_conf  ):
        raise KeyError( " [ERROR] plot_conf does not include 'refp' plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf["refp"].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf["refp"]["default"] }

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
        fig.set__axis()
        fig.save__figure()
        

        
# ========================================================= #
# ===  plot__trajectories                               === #
# ========================================================= #

def plot__trajectories( plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase  = "png/trajectory/{}.png"
    basedir  = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )
    
    if ( plot_conf is None ):
        raise TypeError( " [ERROR] plot_conf is None... " )
    if not( "trajectory" in plot_conf  ):
        raise KeyError( " [ERROR] plot_conf does not include 'trajectory' plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf["trajectory"].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf["trajectory"]["default"] }

    # ------------------------------------------------- #
    # --- [2] plot settings                         --- #
    # ------------------------------------------------- #
    settings = plot_conf["trajectory"]["settings"]
    cmap     = settings.get( "cmap"    , "plasma" )
    nColors  = settings.get( "nColors" , 101      )
    nRandoms = settings.get( "nRandoms", 300      )
    colors   = plt.get_cmap( cmap, nColors )
    files    = plot_conf["files"]
    if ( nRandoms is not None ):
        with open( files["record"], "r" ) as f:
            record = json5.load( f )
        npart  = record["beam.nparticles"]
        if ( nRandoms > npart ):
            raise ValueError( f"nRandoms ({nRandoms}) > number of particles ({npart})")
        pids   = np.random.choice( np.arange(1,npart+1), size=nRandoms, replace=False )
        
    elif ( pids is None ):
        sample = itk.get__particles( recoFile=files["record"], refpFile=files["refp"],
                                     bpmsFile=files["bpms"], steps=[0] )
        pids   = np.array( sorted( sample["pid"].unique() ), dtype=np.int64 )
        
    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        print( f" plotting {key}.... " )
        hconf  = contents["config"]
        config = { **def_conf, **hconf }
        fig    = gp1.gplot1D( config=config, pngFile=pngbase.format( key ) )
        for plot in contents["plots"]:
            for ik,pid in enumerate( tqdm.tqdm(pids) ):
                df       = itk.get__particles( recoFile=files["record"], refpFile=files["refp"], \
                                               bpmsFile=files["bpms"]  , pids    =[pid] )
                plot_ax  = { "xAxis":df[plot["xAxis"]], "yAxis":df[plot["yAxis"]] }
                plot_opt = { key:val for key,val in plot.items() if key not in [ "xAxis","yAxis" ] }
                fig.add__plot( **plot_ax, **plot_opt, color=colors( ik%nColors ), \
                               **contents["option"] )
        fig.set__axis()
        fig.save__figure()
    return()


# ========================================================= #
# ===  plot__postprocessed                              === #
# ========================================================= #

def plot__postprocessed( df=None, plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase = "png/post/{}.png"
    basedir = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )

    if ( plot_conf is None ):
        raise TypeError(" [ERROR] plot_conf is None... ")
    if ( "post" not in plot_conf ):
        raise KeyError(" [ERROR] plot_conf does not include 'post' plot settings...")
    else:
        plots    = { key: val for key, val in plot_conf["post"].items()
                     if key not in ["settings", "default"] }
        def_conf = { **lcf.load__config(), **plot_conf["post"]["default"] }

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
# ===  plot__statistics                                 === #
# ========================================================= #

def plot__statistics( df=None, plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] default settings                      --- #
    # ------------------------------------------------- #
    pngbase    = "png/stat/{}.png"
    basedir    = os.path.dirname( pngbase.format("_") )
    os.makedirs( basedir, exist_ok=True )
    if ( plot_conf is None ):
        raise TypeError( " [ERROR] plot_conf is None... " )
    if not( "stat" in plot_conf  ):
        raise KeyError( " [ERROR] plot_conf does not include 'stat' plot settings..." )
    else:
        plots    = { key:val for key,val in plot_conf["stat"].items()
                     if key not in ["settings","default"] }
        def_conf = { **lcf.load__config(), **plot_conf["stat"]["default"] }
    
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
        fig.set__axis()
        fig.save__figure()

    

# # ========================================================= #
# # ===  plot__poincare2                                  === #
# # ========================================================= #

# def plot__poincare2( bpmsFile="impactx/diags/openPMD/bpm.h5", step=0 ):

#     mm, mrad  = 1.0e-3, 1.0e-3
#     cv, nsec  = 2.99792458e8, 1.0e-9

#     # ------------------------------------------------- #
#     # --- [1] plot config                           --- #
#     # ------------------------------------------------- #
#     config   = lcf.load__config()
#     def_conf = {
#         "figure.size"        : [4.5,4.5],
#         "figure.position"    : [ 0.18, 0.18, 0.92, 0.92 ],
#         "ax1.x.minor.nticks" : 1, 
#         "plot.linestyle"     : "none", 
#         "plot.marker"        : "o",
#         "plot.markersize"    : 0.2,
#         "legend.fontsize"    : 9.0, 
#     }
#     def_conf = { **config, **def_conf }

#     config_ = {
#         "xp-px" : {
#             "figure.pngFile"     : "png/poincare__xp-px_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -axpMax, "max":axpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apxMax, "max":apxMax, "num":11 },
#             "ax1.x.label"        : "x (mm)",
#             "ax1.y.label"        : "x' (mrad)",
#         },
#         "yp-py" : {
#             "figure.pngFile"     : "png/poincare__yp-py_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -aypMax, "max":+aypMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apyMax, "max":+apyMax, "num":11 },
#             "ax1.x.label"        : "y (mm)",
#             "ax1.y.label"        : "y' (mrad)",
#         },
#         "tp-pt" : {
#             "figure.pngFile"     : "png/poincare__tp-pt_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -atpMax, "max":+atpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -aptMax, "max":+aptMax, "num":11 },
#             "ax1.x.label"        : "t (mm)",
#             "ax1.y.label"        : "t' (mrad)",
#         },
#     }
    
#     # ------------------------------------------------- #
#     # --- [2] plot poinncare                        --- #
#     # ------------------------------------------------- #
#     with h5py.File( bpmsFile, "r" ) as bpms:
#         for chunk in itk.load__bpms( bpms=bpms, step=step ):
#             chunk["xp"] = chunk["xp"] / mm
#             chunk["yp"] = chunk["yp"] / mm
#             chunk["tp"] = chunk["tp"] / mm
#             chunk["px"] = chunk["px"] / mrad
#             chunk["py"] = chunk["py"] / mrad
#             chunk["pt"] = chunk["pt"] / mrad
#             axpMax = np.max( [ np.abs( np.min( chunk["xp"] ) ), np.abs( np.max( chunk["xp"] ) ) ] )
#             aypMax = np.max( [ np.abs( np.min( chunk["yp"] ) ), np.abs( np.max( chunk["yp"] ) ) ] )
#             atpMax = np.max( [ np.abs( np.min( chunk["tp"] ) ), np.abs( np.max( chunk["tp"] ) ) ] )
#             apxMax = np.max( [ np.abs( np.min( chunk["px"] ) ), np.abs( np.max( chunk["px"] ) ) ] )
#             apyMax = np.max( [ np.abs( np.min( chunk["py"] ) ), np.abs( np.max( chunk["py"] ) ) ] )
#             aptMax = np.max( [ np.abs( np.min( chunk["pt"] ) ), np.abs( np.max( chunk["pt"] ) ) ] )

#         for key,contents in plots.items():
#             if ( plot_conf is not None ):
#                 config = { **config, **plot_conf[key]["config"] }
#         fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
#         for content in contents:
#             fig.add__plot( **content )
#         fig.set__legend()
#         fig.save__figure()

#         for key 
#         config = { **def_conf, **config_[key] }
#         fig = gp1.gplot1D( config=config )
#         for content in contents:
#             fig.add__scatter( **content )
#         fig.set__axis()
#         fig.save__figure()


# # ========================================================= #
# # ===  plot__statistics.py                              === #
# # ========================================================= #
# def plot__statistics( inpFile=None, pngDir="png/", plot_conf=None  ):

#     ylabels_ = { "x-range"        : "x [m]"                 ,
#                  "y-range"        : "y [m]"                 ,
#                  "t-range"        : "t [m]"                 ,
#                  "px-range"       : "px"                    ,
#                  "py-range"       : "py"                    ,
#                  "pt-range"       : "pt"                    ,
#                  "sigma_xyt"      : "RMS beam size [m]"     , 
#                  "sigma_pxyt"     : "Momentum"              ,
#                  "alpha_xyt"      : r"$\alpha$"             ,
#                  "beta_xyt"       : r"$\beta$ [m]"          ,
#                  "dispersion_xy"  : r"$Dispersion$"         ,
#                  "dispersion_pxy" : r"$Dispersion_p$"       ,
#                  "emittance_xyt"  : r"$\epsilon$ [m rad]"   ,
#                  "emittance_xytn" : r"$\epsilon_n$ [m  rad]",
#                  "charge_C"       : "Charge [C]"            , 
#                 }
#     THRESHOLD = 1.e30
    
#     # ------------------------------------------------- #
#     # --- [1] load .stat file                       --- #
#     # ------------------------------------------------- #
#     data    = pd.read_csv( inpFile, sep=r"\s+" )
#     data    = data.where( data.abs() <= THRESHOLD, np.nan )
#     ylabels = { key:key for key in data.keys() }
#     ylabels = { **ylabels, **ylabels_ }

#     nparticle_threshold = 0.10
#     if ( nparticle_threshold ):
#         fraction = ( data["charge_C"] ) / ( data["charge_C"][0] )
#         data     = data.where( fraction >= nparticle_threshold, np.nan )
    
#     # ------------------------------------------------- #
#     # --- [2] plot config                           --- #
#     # ------------------------------------------------- #
#     config   = lcf.load__config()
#     config_  = {
#         "figure.size"        : [10.5,3.5],
#         "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
#         "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
#         "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
#         "ax1.x.label"        : "s [m]",
#         "ax1.x.minor.nticks" : 1, 
#         "plot.marker"        : "o",
#         "plot.markersize"    : 2.0,
#         "legend.fontsize"    : 8.0, 
#     }
#     config = { **config, **config_ }
#     if ( plot_conf is not None ):
#         config = { **config, **plot_conf }        


#     # ------------------------------------------------- #
#     # --- [3] xyt range ( mean, min, max ) graph    --- #
#     # ------------------------------------------------- #
#     for xyt in ["x","y","t", "px","py","pt"]:
#         config_  = {
#             "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}-range.png".format( xyt ) ), 
#             "ax1.y.label"    : ylabels["{}-range".format( xyt )],
#         }
#         config = { **config, **config_ }
#         fig    = gp1.gplot1D( config=config )
#         fig.add__plot( xAxis=data["s"], yAxis=data["mean_"+xyt], label="Mean" )
#         fig.add__plot( xAxis=data["s"], yAxis=data["min_"+xyt] , label="Min"  )
#         fig.add__plot( xAxis=data["s"], yAxis=data["max_"+xyt] , label="Max"  )
#         fig.set__axis()
#         fig.set__legend()
#         fig.save__figure()

#     # ------------------------------------------------- #
#     # --- [4] statistics graph                      --- #
#     # ------------------------------------------------- #
#     for stat in [ "sigma_{}", "sigma_p{}", "alpha_{}", "beta_{}", \
#                   "emittance_{}", "emittance_{}n" ]:
#         gname    = stat.format( "xyt" )
#         config_  = {
#             "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
#             "ax1.y.label"    : ylabels[gname],
#         }
#         config = { **config, **config_ }
#         fig    = gp1.gplot1D( config=config )
#         fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
#         fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
#         fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("t")], label=stat.format("t")  )
#         fig.set__axis()
#         fig.set__legend()
#         fig.save__figure()

#     # ------------------------------------------------- #
#     # --- [5] dispersion                            --- #
#     # ------------------------------------------------- #
#     for stat in [ "dispersion_{}", "dispersion_p{}", ]:
#         gname    = stat.format( "xy" )
#         config_  = {
#             "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
#             "ax1.y.label"    : ylabels[gname],
#         }
#         config = { **config, **config_ }
#         fig    = gp1.gplot1D( config=config )
#         fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
#         fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
#         fig.set__axis()
#         fig.set__legend()
#         fig.save__figure()

#     # ------------------------------------------------- #
#     # --- [5] charge C                              --- #
#     # ------------------------------------------------- #
#     config_  = {
#         "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( "charge_C" ) ), 
#         "ax1.y.label"    : ylabels["charge_C"],
#     }
#     config = { **config, **config_ }
#     fig    = gp1.gplot1D( config=config )
#     fig.add__plot( xAxis=data["s"], yAxis=data["charge_C"] )
#     fig.set__axis()
#     fig.save__figure()


# # ========================================================= #
# # ===  plot__poincare                                   === #
# # ========================================================= #

# def plot__poincare( bpms=None, step=None ):

#     mm, mrad  = 1.0e-3, 1.0e-3
#     cv, nsec  = 2.99792458e8, 1.0e-9
    
#     if ( bpms is None ): sys.exit( "[plot_poincare] bpms == ???" )
#     if ( step is None ): step = sorted( list( set(bpms["step"]) ) )[0]
#     if ( type(bpms) is str ):
#         bpms = h5py.File( bpms, "r" )

#     # ------------------------------------------------- #
#     # --- [1] prepare variables                     --- #
#     # ------------------------------------------------- #
#     if ( type(bpms) is pd.DataFrame ):
#         bpms_        = bpms.loc[ bpms["step"] == step ].copy()
#         bpms_["xp"]  = bpms_["xp"] / mm
#         bpms_["yp"]  = bpms_["yp"] / mm
#         bpms_["tp"]  = bpms_["tp"] / mm
#         bpms_["px"]  = bpms_["px"] / mrad
#         bpms_["py"]  = bpms_["py"] / mrad
#         bpms_["pt"]  = bpms_["pt"] / mrad
#         bpms_["dt"]  = bpms_["dt"] / nsec  # [ns]
#         bpms_["dEk"] = bpms_["dEk"]        # [MeV]

#     axpMax = np.max( [ np.abs( bpms_["xp"].min() ), np.abs( bpms_["xp"].max() ) ] )
#     aypMax = np.max( [ np.abs( bpms_["yp"].min() ), np.abs( bpms_["yp"].max() ) ] )
#     atpMax = np.max( [ np.abs( bpms_["tp"].min() ), np.abs( bpms_["tp"].max() ) ] )
#     apxMax = np.max( [ np.abs( bpms_["px"].min() ), np.abs( bpms_["px"].max() ) ] )
#     apyMax = np.max( [ np.abs( bpms_["py"].min() ), np.abs( bpms_["py"].max() ) ] )
#     aptMax = np.max( [ np.abs( bpms_["pt"].min() ), np.abs( bpms_["pt"].max() ) ] )
#     adtMax = np.max( [ np.abs( bpms_["dt"].min() ), np.abs( bpms_["dt"].max() ) ] )
#     adEMax = np.max( [ np.abs( bpms_["dEk"].min() ), np.abs( bpms_["dEk"].max() ) ] )
    
#     # ------------------------------------------------- #
#     # --- [2] plot config                           --- #
#     # ------------------------------------------------- #
#     config   = lcf.load__config()
#     def_conf = {
#         "figure.size"        : [4.5,4.5],
#         "figure.position"    : [ 0.18, 0.18, 0.92, 0.92 ],
#         "ax1.x.minor.nticks" : 1, 
#         "plot.linestyle"     : "none", 
#         "plot.marker"        : "o",
#         "plot.markersize"    : 0.2,
#         "legend.fontsize"    : 9.0, 
#     }
#     def_conf = { **config, **def_conf }

#     config_ = {
#         "xp-px" : {
#             "figure.pngFile"     : "png/poincare__xp-px_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -axpMax, "max":axpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apxMax, "max":apxMax, "num":11 },
#             "ax1.x.label"        : "x (mm)",
#             "ax1.y.label"        : "x' (mrad)",
#         },
#         "yp-py" : {
#             "figure.pngFile"     : "png/poincare__yp-py_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -aypMax, "max":+aypMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apyMax, "max":+apyMax, "num":11 },
#             "ax1.x.label"        : "y (mm)",
#             "ax1.y.label"        : "y' (mrad)",
#         },
#         "tp-pt" : {
#             "figure.pngFile"     : "png/poincare__tp-pt_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -atpMax, "max":+atpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -aptMax, "max":+aptMax, "num":11 },
#             "ax1.x.label"        : "t (mm)",
#             "ax1.y.label"        : "t' (mrad)",
#         },
#         "dt-dE" : {
#             "figure.pngFile"     : "png/poincare__dt-dE_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -adtMax, "max":+adtMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -adEMax, "max":+adEMax, "num":11 },
#             "ax1.x.label"        : "dt (ns)",
#             "ax1.y.label"        : "dE (MeV)",
#         },
#     }
#     # ------------------------------------------------- #
#     # --- [3] plots settings                        --- #
#     # ------------------------------------------------- #
#     plots      = {
#         "xp-px": [ { "xAxis":bpms_["xp"], "yAxis":bpms_["px"] , "density":True }, ], \
#         "yp-py": [ { "xAxis":bpms_["yp"], "yAxis":bpms_["py"] , "density":True }, ], \
#         "tp-pt": [ { "xAxis":bpms_["tp"], "yAxis":bpms_["pt"] , "density":True }, ], \
#         "dt-dE": [ { "xAxis":bpms_["dt"], "yAxis":bpms_["dEk"], "density":True }, ], \
#     }
    
#     # ------------------------------------------------- #
#     # --- [3] plot                                  --- #
#     # ------------------------------------------------- #
#     for key,contents in plots.items():
#         config = { **def_conf, **config_[key] }
#         fig = gp1.gplot1D( config=config )
#         for content in contents:
#             fig.add__scatter( **content )
#         fig.set__axis()
#         fig.save__figure()



# # ========================================================= #
# # ===  plot__refparticle.py                             === #
# # ========================================================= #
# def plot__refparticle( refpFile=None, pngDir="png/refp/", plot_conf=None  ):

#     ylabels  = { "beta"       : r"$\beta$"        ,
#                  "gamma"      : r"$\gamma$"       ,
#                  "beta_gamma" : r"$\beta \gamma$" ,
#                  "x"          : r"$x$ [m]"        ,
#                  "y"          : r"$y$ [m]"        ,
#                  "z"          : r"$z$ [m]"        ,
#                  "t"          : r"$ct$ [m]"       ,
#                  "px"         : r"$p_x$ [rad]"    ,
#                  "py"         : r"$p_y$ [rad]"    ,
#                  "pz"         : r"$p_z$ [rad]"    ,
#                  "pt"         : r"$p_t$ [rad]"    ,
#                 }
#     def_config = { }
#     THRESHOLD = 1e32
    
    # ------------------------------------------------- #
    # # --- [1] load .stat file                       --- #
    # # ------------------------------------------------- #
    # data       = pd.read_csv( refpFile, sep=r"\s+" )
    # data       = data.where( data.abs() <= THRESHOLD, np.nan )
    # xAxis      = data["s"]
    # os.makedirs( pngDir, exist_ok=True )
    
    # # ------------------------------------------------- #
    # # --- [2] plot config                           --- #
    # # ------------------------------------------------- #
    # plot_conf_ = plot_conf.get( "refp", {} )
    # config     = { **lcf.load__config(), **def_config }

    # # ------------------------------------------------- #
    # # --- [3] xyt range ( mean, min, max ) graph    --- #
    # # ------------------------------------------------- #
    # for key,hylabel in ylabels.items():
    #     pkey    = "s-{}".format( key )
    #     hconfig = { "figure.pngFile" : os.path.join( pngDir, f"refp__{key}.png" ), 
    #                 "ax1.y.label"    : hylabel,
    #                }
    #     config  = { **config, **hconfig }
    #     if key in plot_conf_:
    #         config = { **config, **plot_conf_[key] }
    #     fig     = gp1.gplot1D( config=config )
    #     fig.add__plot( xAxis=xAxis, yAxis=data[key] )
    #     fig.set__axis()
    #     fig.save__figure()
        



# # ========================================================= #
# # ===  plot__postprocessed                              === #
# # ========================================================= #

# def plot__postprocessed( df=None, plot_conf=None ):

#     # ------------------------------------------------- #
#     # --- [1] load post processed / set variables   --- #
#     # ------------------------------------------------- #
    
#     plots      = {
#         "post__s-Ek_ref": [ { "xAxis":data["s"], "yAxis":data["Ek_ref"], \
#                               "label":"kinetic", "ylabel":"Energy (MeV)" } ],
#         "post__s-Ek"    : [ { "xAxis":data["s"], "yAxis":data["Ek_min"], \
#                               "label":r"$min(E_k)$", "ylabel":"Energy (MeV)" }, \
#                             { "xAxis":data["s"], "yAxis":data["Ek_avg"], \
#                               "label":r"$avg(E_k)$", "ylabel":"Energy (MeV)" }, \
#                             { "xAxis":data["s"], "yAxis":data["Ek_max"], \
#                               "label":r"$max(E_k)$", "ylabel":"Energy (MeV)" }, \
#                            ], \
#         "post__s-dphi"  : [ { "xAxis":data["s"], "yAxis":data["dphi_min"], "label":"min.", },
#                             { "xAxis":data["s"], "yAxis":data["dphi_avg"], "label":"avg.", },
#                             { "xAxis":data["s"], "yAxis":data["dphi_max"], "label":"max.", 
#                               "ylabel":r"$\Delta \phi \mathrm{(deg)}$" } ],
#         "post__s-dp_p0" : [ { "xAxis":data["s"], "yAxis":data["dp/p_min"], "label":"min.", },
#                             { "xAxis":data["s"], "yAxis":data["dp/p_avg"], "label":"avg.", },
#                             { "xAxis":data["s"], "yAxis":data["dp/p_max"], "label":"max.", \
#                               "ylabel":r"$\Delta p/p \ \mathrm{(\%)}$" } ],
#         "post__s-dE_E0" : [ { "xAxis":data["s"], "yAxis":data["dE/E_min"], "label":"min.", },
#                             { "xAxis":data["s"], "yAxis":data["dE/E_avg"], "label":"avg.", },
#                             { "xAxis":data["s"], "yAxis":data["dE/E_max"], "label":"max.", \
#                               "ylabel":r"$\Delta E/E \ \mathrm{(\%)}$" } ],
#         "post__s-sigma_phi"     : [ { "xAxis":data["s"], "yAxis":data["dphi_rms"], "label":"RMS",
#                                       "ylabel":r"$\sigma_{\phi} \ \mathrm{(deg)}$" } ],
#         "post__s-sigma_dp_p"    : [ { "xAxis":data["s"], "yAxis":data["dp/p_rms"], "label":"RMS",
#                                       "ylabel":r"$\sigma_{\Delta p/p} \ \mathrm{(\%)}$" } ],
#         "post__s-sigma_dE_E0"   : [ { "xAxis":data["s"], "yAxis":data["dE/E_rms"], "label":"RMS",
#                                       "ylabel":r"$\sigma_{\Delta E/E} \ \mathrm{(\%)}$" } ],
#         "post__s-transmission"  : [ { "xAxis":data["s"], "yAxis":data["transmission"], \
#                                       "label":"Transmission", 
#                                       "ylabel":"Transmission (%)", "yRange":[ 0.0, 120.0 ] } ],
#         "post__s-max_sigma_xy"  : [ { "xAxis":data["s"], "yAxis":data["max/sigma_x"], "label":"x",
#                                       "ylabel":r"$max(|x|)/\sigma_x, max(|y|)/\sigma_y$"}, \
#                                     { "xAxis":data["s"], "yAxis":data["max/sigma_y"], "label":"y"}],
#         "post__s-max_sigma_t"   : [ { "xAxis":data["s"], "yAxis":data["max/sigma_t"], "label":"t",
#                                       "ylabel":r"$max(|t|)/\sigma_t$" } ], 
#         "post__s-max_sigma_dphi": [ { "xAxis":data["s"], "yAxis":data["max/sigma_dphi"],
#                                       "label":r"$\Delta \phi$",
#                                       "ylabel":r"$max(|\Delta \phi|)/\sigma_{\Delta \phi}$" } ],
#         "post__s-corr__xp-yp"   : [ { "xAxis":data["s"], "yAxis":data["xp-yp"]*100.0, \
#                                       "label":r"$<x,y>$", \
#                                       "ylabel":r"$cov(x,y)/\sigma_x \sigma_y \ (\%)$" } ],
#         "post__s-corr__xp-tp"   : [ { "xAxis":data["s"], "yAxis":data["xp-tp"]*100.0, \
#                                       "label":r"$<x,t>$", \
#                                       "ylabel":r"$cov(x,t)/\sigma_x \sigma_t \ (\%)$" } ],
#         "post__s-corr__xp-px"   : [ { "xAxis":data["s"], "yAxis":data["xp-px"]*100.0, \
#                                       "label":r"$<x,p_x>$", \
#                                       "ylabel":r"$cov(x,p_x)/\sigma_x \sigma_{p_x}$ \ (\%)" } ],
#         "post__s-corr__xp-py"   : [ { "xAxis":data["s"], "yAxis":data["xp-py"]*100.0, \
#                                       "label":r"$<x,p_y>$", \
#                                       "ylabel":r"$cov(x,p_y)/\sigma_x \sigma_{p_y}$ \ (\%)" } ],
#         "post__s-corr__xp-pt"   : [ { "xAxis":data["s"], "yAxis":data["xp-pt"]*100.0, \
#                                       "label":r"$<x,p_t>$", \
#                                       "ylabel":r"$cov(x,p_t)/\sigma_x \sigma_{p_t}$ \ (\%)" } ],
#         "post__s-corr__tp-pt"   : [ { "xAxis":data["s"], "yAxis":data["tp-pt"]*100.0, \
#                                       "label":r"$<t,p_t>$", \
#                                       "ylabel":r"$cov(t,p_t)/\sigma_t \sigma_{p_t}$ \ (\%)" } ],

#     }
#     xmin, xmax = np.min( data["s"] ), np.max( data["s"] )
    
#     # ------------------------------------------------- #
#     # --- [2] plot config                           --- #
#     # ------------------------------------------------- #
#     config   = lcf.load__config()
#     config_  = {
#         "figure.size"        : [9.0,3.0],
#         "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
#         "ax1.x.range"        : { "auto":True, "min": xmin, "max":xmax, "num":8 },
#         "ax1.y.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":6 },
#         "ax1.x.label"        : "s (m)",
#         "ax1.x.minor.nticks" : 1, 
#         "plot.marker"        : "o",
#         "plot.markersize"    : 2.0,
#         "legend.fontsize"    : 8.0, 
#     }
#     config = { **config, **config_ }
#     if ( plot_conf is not None ):
#         config = { **config, **plot_conf }

#     # ------------------------------------------------- #
#     # --- [3] plot                                  --- #
#     # ------------------------------------------------- #
#     for key,contents in plots.items():
#         fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
#         for content in contents:
#             fig.add__plot( **content )
#         fig.set__legend()
#         fig.save__figure()

