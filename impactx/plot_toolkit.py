import os, sys, json5
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


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
# ===  plot__refparticle.py                             === #
# ========================================================= #
def plot__refparticle( inpFile=None, pngDir="png/", plot_conf=None  ):

    ylabels  = { "beta"       : r"$\beta$"        ,
                 "gamma"      : r"$\gamma$"       ,
                 "beta_gamma" : r"$\beta \gamma$" ,
                 "x"          : r"$x$ [m]"        ,
                 "y"          : r"$y$ [m]"        ,
                 "z"          : r"$z$ [m]"        ,
                 "t"          : r"$ct$ [m]"       ,
                 "px"         : r"$p_x$ [rad]"    ,
                 "py"         : r"$p_y$ [rad]"    ,
                 "pz"         : r"$p_z$ [rad]"    ,
                 "pt"         : r"$p_t$ [rad]"    ,
                }
    THRESHOLD = 1e32
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data       = pd.read_csv( inpFile, sep=r"\s+" )
    data       = data.where( data.abs() <= THRESHOLD, np.nan )
    xAxis      = data["s"]
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config  = lcf.load__config()
    config_ = {
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
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        

    # ------------------------------------------------- #
    # --- [3] xyt range ( mean, min, max ) graph    --- #
    # ------------------------------------------------- #
    for key,hylabel in ylabels.items():
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "refp__s-{0}.png".format( key ) ), 
            "ax1.y.label"    : hylabel,
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=xAxis, yAxis=data[key] )
        fig.set__axis()
        fig.save__figure()


# ========================================================= #
# ===  plot__statistics.py                              === #
# ========================================================= #
def plot__statistics( inpFile=None, pngDir="png/", plot_conf=None  ):

    ylabels_ = { "x-range"        : "x [m]"                 ,
                 "y-range"        : "y [m]"                 ,
                 "t-range"        : "t [m]"                 ,
                 "px-range"       : "px"                    ,
                 "py-range"       : "py"                    ,
                 "pt-range"       : "pt"                    ,
                 "sigma_xyt"      : "RMS beam size [m]"     , 
                 "sigma_pxyt"     : "Momentum"              ,
                 "alpha_xyt"      : r"$\alpha$"             ,
                 "beta_xyt"       : r"$\beta$ [m]"          ,
                 "dispersion_xy"  : r"$Dispersion$"         ,
                 "dispersion_pxy" : r"$Dispersion_p$"       ,
                 "emittance_xyt"  : r"$\epsilon$ [m rad]"   ,
                 "emittance_xytn" : r"$\epsilon_n$ [m  rad]",
                 "charge_C"       : "Charge [C]"            , 
                }
    THRESHOLD = 1.e30
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data    = pd.read_csv( inpFile, sep=r"\s+" )
    data    = data.where( data.abs() <= THRESHOLD, np.nan )
    ylabels = { key:key for key in data.keys() }
    ylabels = { **ylabels, **ylabels_ }

    nparticle_threshold = 0.10
    if ( nparticle_threshold ):
        fraction = ( data["charge_C"] ) / ( data["charge_C"][0] )
        data     = data.where( fraction >= nparticle_threshold, np.nan )
    
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
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        


    # ------------------------------------------------- #
    # --- [3] xyt range ( mean, min, max ) graph    --- #
    # ------------------------------------------------- #
    for xyt in ["x","y","t", "px","py","pt"]:
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}-range.png".format( xyt ) ), 
            "ax1.y.label"    : ylabels["{}-range".format( xyt )],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data["mean_"+xyt], label="Mean" )
        fig.add__plot( xAxis=data["s"], yAxis=data["min_"+xyt] , label="Min"  )
        fig.add__plot( xAxis=data["s"], yAxis=data["max_"+xyt] , label="Max"  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] statistics graph                      --- #
    # ------------------------------------------------- #
    for stat in [ "sigma_{}", "sigma_p{}", "alpha_{}", "beta_{}", \
                  "emittance_{}", "emittance_{}n" ]:
        gname    = stat.format( "xyt" )
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
            "ax1.y.label"    : ylabels[gname],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("t")], label=stat.format("t")  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] dispersion                            --- #
    # ------------------------------------------------- #
    for stat in [ "dispersion_{}", "dispersion_p{}", ]:
        gname    = stat.format( "xy" )
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
            "ax1.y.label"    : ylabels[gname],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] charge C                              --- #
    # ------------------------------------------------- #
    config_  = {
        "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( "charge_C" ) ), 
        "ax1.y.label"    : ylabels["charge_C"],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["charge_C"] )
    fig.set__axis()
    fig.save__figure()


# ========================================================= #
# ===  plot__trajectories.py                            === #
# ========================================================= #

def plot__trajectories( hdf5File=None, refpFile=None, pids=None, random_choice=None, \
                        cmap="plasma", nColors=128, pngDir="png/", plot_conf=None ):

    ylabels  = { "xp":r"$x$ [mm]"  , "yp":r"$y$ [mm]"  , "tp":r"$t$ [mm]", \
                 "px":r"$p_x$ [mrad]", "py":r"$p_y$ [mrad]", "pz":r"$p_z$ [mrad]" }
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    pinfo    = load__impactHDF5( inpFile=hdf5File, \
                                 pids=pids, random_choice=random_choice )
    rinfo    = pd.read_csv( refpFile, sep=r"\s+"  )
    ref_s    = rinfo["s"]
    colors   = plt.get_cmap( cmap, nColors )
    if ( pids is None ):
        pids = np.array( pinfo["pid"].unique(), dtype=np.int64 )
    
    # ------------------------------------------------- #
    # --- [2] configuration                         --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.12, 0.97, 0.95 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "plot.marker"        : "none",
        "plot.linestyle"     : "-", 
        "plot.markersize"    : 0.2,
    }
    config   = { **config, **config_ }
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        

    pngFile  = os.path.join( pngDir, "traj__s-{}.png" )

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for obj in [ "xp","yp","tp","px","py","pz" ]:
        fig = gp1.gplot1D( config=config, pngFile=pngFile.format( obj ) )
        config["ax1.y.label"] = ylabels[obj]
        for ik,pid in enumerate(tqdm.tqdm(pids)):
            traj = ( pinfo[ pinfo["pid"] == pid ] )[obj].values
            if ( len(traj) == 0 ): continue
            xAxis  = ref_s[:(len(traj))]
            hcolor = colors( ik%nColors )
            fig.add__plot( xAxis=xAxis, yAxis=traj, color=hcolor )
        fig.set__axis()
        fig.save__figure()
    return()


# ========================================================= #
# ===  plot__postProcessed                              === #
# ========================================================= #

def plot__postProcessed( paramsFile="dat/parameters.json", \
                         postFile="impactx/diags/postProcessed_beam.csv", plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] load post processed / set variables   --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    data       = pd.read_csv( postFile )
    plots      = {
        "post__s-Ek":{ "xAxis":data["s_refp"], "yAxis":data["Ek"], "ylabel":"Energy (MeV)" },
    }
    xmin, xmax = np.min( data["s_refp"] ), np.max( data["s_refp"] )
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": xmin, "max":xmax, "num":11 },
        "ax1.y.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
        fig.add__plot( **contents )
        fig.set__axis()
        fig.save__figure()
            
